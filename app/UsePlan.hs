{-# LANGUAGE OverloadedStrings #-}

module UsePlan (run, readRational) where

import Control.Applicative ((<|>))
import Control.Exception (try)
import Control.Monad (join, unless)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value (Null), object, (.=))
import Data.Aeson.Types (Pair)
import Data.Char (isDigit)
import Data.Foldable (toList)
import Data.Maybe (isJust)
import Envelope (Problem (..), rational, refuse, succeed)
import Invar.Artifact qualified as Artifact
import Invar.Use qualified as U
import Invar.Use.Confidence qualified as Confidence
import Invar.Use.Plan qualified as Plan
import Numeric.Natural (Natural)
import Options qualified as O
import System.Console.GetOpt (OptDescr)
import System.Directory (doesFileExist, makeAbsolute, withCurrentDirectory)
import System.Exit (ExitCode)
import System.FilePath (takeDirectory)
import Text.Read (readMaybe)
import UseInput (readContract)
import UseRuns qualified

format :: String
format = "invar-use-plan-v1"

run :: [String] -> IO ()
run supplied = do
    fields <- either (\problem -> refuse format [Problem "missing-argument" "argv" problem]) pure (O.parse options supplied)
    contractPath <- required fields "contract"
    unitsText <- required fields "units"
    count <- maybe (refuse format [Problem "invalid-value" "argv:--units" "The planned unit count must be a natural number"]) pure (readMaybe unitsText :: Maybe Natural)
    (contract, contractBytes) <- readContract format "contract" contractPath
    referenceVariance <- parameter fields "reference-variance"
    increaseVariance <- parameter fields "increase-variance"
    referenceMean <- parameter fields "assumed-reference-loss"
    increaseMean <- parameter fields "assumed-increase"
    probe <- case (O.optional fields "probe-contract", O.optional fields "probe") of
        (Nothing, Nothing) -> pure Nothing
        (Just probeContract, Just runs) -> Just <$> readProbe probeContract runs
        _ -> refuse format [Problem "missing-argument" "argv:--probe" "A probe needs both --probe-contract and --probe"]
    let fromProbe select = select <$> probe
        useMeans = isJust (O.optional fields "assume-probe-means")
        conflicts =
            [ Problem "invalid-value" ("argv:--" ++ name) "This value is both assumed explicitly and taken from the probe"
            | (name, stated, probed) <- [("reference-variance", referenceVariance, fromProbe probeReferenceVariance), ("increase-variance", increaseVariance, fromProbe probeIncreaseVariance)] ++ [(name, stated, probed) | useMeans, (name, stated, probed) <- [("assumed-reference-loss", referenceMean, fromProbe probeReferenceMean), ("assumed-increase", increaseMean, fromProbe probeIncreaseMean)]]
            , isJust stated && isJust probed
            ]
    unless (null conflicts) (refuse format conflicts)
    let pick stated probed = stated <|> join probed
        assumptions =
            Plan.Assumptions
                count
                (pick referenceVariance (fromProbe probeReferenceVariance))
                (pick increaseVariance (fromProbe probeIncreaseVariance))
                (pick referenceMean (if useMeans then fromProbe probeReferenceMean else Nothing))
                (pick increaseMean (if useMeans then fromProbe probeIncreaseMean else Nothing))
        estimate = Plan.estimate (U.lossRequirement (U.criterion contract)) assumptions
    succeed format "estimated" "A conditional estimate was computed from the stated assumptions." "Evidence of any kind; it cannot enter a finding or an admission." (["contract" .= contractPath, "contract_sha256" .= Artifact.hex (SHA256.hash contractBytes)] ++ encodeEstimate estimate)
  where
    required fields name = maybe (refuse format [Problem "missing-argument" ("argv:--" ++ name) ("--" ++ name ++ " is required")]) pure (O.optional fields name)
    parameter fields name = case O.optional fields name of
        Nothing -> pure Nothing
        Just text -> maybe (refuse format [Problem "invalid-value" ("argv:--" ++ name) "Expected an exact rational such as 1/40 or a decimal such as 0.025"]) (pure . Just . (`Plan.Parameter` Plan.Assumption)) (readRational text)

data Probe = Probe {probeReferenceVariance, probeIncreaseVariance, probeReferenceMean, probeIncreaseMean :: Maybe Plan.Parameter}

readProbe :: FilePath -> FilePath -> IO Probe
readProbe contractPath runsPath = do
    (contract, bytes) <- readContract format "probe-contract" contractPath
    present <- doesFileExist runsPath
    if present then pure () else refuse format [Problem "artifact-missing" "argv:--probe" (runsPath ++ " does not exist")]
    absolute <- makeAbsolute runsPath
    observed <- try (withCurrentDirectory (takeDirectory absolute) (UseRuns.observe absolute contract)) :: IO (Either ExitCode (Either String U.Observed))
    case observed of
        Right (Right value) -> pure (summarize (Artifact.hex (SHA256.hash bytes)) value)
        Right (Left problem) -> refuse format [Problem "artifact-invalid" ("artifact:" ++ runsPath) problem]
        Left _ -> refuse format [Problem "artifact-invalid" ("artifact:" ++ runsPath) "The probe records could not be read under the probe contract"]
  where
    summarize digest observed =
        let count = fromIntegral (length (U.units observed))
            source = Plan.Probe count digest
            statistic metric = toList <$> U.measurements metric observed
            variance metric = statistic metric >>= Confidence.sampleVariance >>= \value -> Just (Plan.Parameter value source)
            average metric = statistic metric >>= \values -> Just (Plan.Parameter (sum values / fromIntegral (length values)) source)
         in Probe (variance U.ReferenceLoss) (variance U.LossIncrease) (average U.ReferenceLoss) (average U.LossIncrease)

readRational :: String -> Maybe Rational
readRational text = case break (== '/') text of
    (whole, '/' : below) -> do
        top <- readMaybe whole :: Maybe Integer
        bottom <- readMaybe below :: Maybe Integer
        if bottom > 0 then Just (fromInteger top / fromInteger bottom) else Nothing
    _ -> decimal text
  where
    decimal ('-' : rest) = negate <$> decimal rest
    decimal digits = case break (== '.') digits of
        (whole, '.' : fraction) | valid whole && valid fraction -> Just (fromInteger (read (whole ++ fraction)) / 10 ^ length fraction)
        (whole, "") | valid whole -> Just (fromInteger (read whole))
        _ -> Nothing
    valid part = not (null part) && all isDigit part

encodeEstimate :: Plan.PlanEstimate -> [Pair]
encodeEstimate (Plan.NotApplicable reason) = ["standard" .= ("not applicable" :: String), "bound" .= ("not applicable: " ++ reason)]
encodeEstimate (Plan.Estimated standard count bounds) =
    [ "standard" .= (case standard of Plan.Hoeffding -> "population_hoeffding"; Plan.Bernstein -> "population_bernstein_mp2009" :: String)
    , "units" .= count
    , "search_limit" .= Plan.searchLimit
    , "bounds" .= map bound bounds
    ]
  where
    bound value =
        object
            [ "metric" .= show (Plan.boundMetric value)
            , "range" .= rational (Plan.boundRange value)
            , "alpha" .= rational (Plan.boundAlpha value)
            , "ceiling" .= rational (Plan.boundCeiling value)
            , "variance" .= maybe Null parameter (Plan.boundVariance value)
            , "assumed_mean" .= maybe Null parameter (Plan.boundMean value)
            , "zero_assumed" .= (fmap Plan.parameterValue (Plan.boundMean value) == Just 0)
            , "width" .= maybe (object ["missing" .= ("a variance assumption or probe is required" :: String)]) rational (Plan.boundWidth value)
            , "feasible" .= Plan.boundFeasible value
            , "sample_size" .= maybe Null search (Plan.boundSearch value)
            ]
    parameter (Plan.Parameter value source) = object (("value" .= rational value) : origin source)
    origin Plan.Assumption = ["source" .= ("assumption" :: String)]
    origin (Plan.Probe units digest) = ["source" .= ("probe" :: String), "statistic" .= ("computed over probe units" :: String), "units" .= units, "probe_contract_sha256" .= digest, "role" .= ("planning assumption" :: String)]
    search (Plan.Found n) = object ["result" .= ("found" :: String), "units" .= n, "note" .= ("smallest N within the searched range" :: String)]
    search (Plan.NotFoundWithin limit) = object ["result" .= ("not_found_within_limit" :: String), "limit" .= limit]
    search Plan.NoFiniteSolution = object ["result" .= ("no_finite_solution" :: String), "reason" .= ("the assumed mean exceeds a ceiling below one, so every bound exceeds it" :: String)]

options :: [OptDescr (String, String)]
options = O.descriptions [("contract", "Use contract to plan for"), ("units", "Planned number of independent units"), ("reference-variance", "Assumed unit variance of the reference loss"), ("increase-variance", "Assumed unit variance of the paired loss increase"), ("assumed-reference-loss", "Assumed mean reference loss"), ("assumed-increase", "Assumed mean loss increase"), ("probe-contract", "Contract the probe was recorded under"), ("probe", "Runs file of the probe, read relative to its own directory"), ("assume-probe-means", "Use the probe means as planning assumptions (value: yes)")]
