{-# LANGUAGE OverloadedStrings #-}

module Use (run) where

import Data.Aeson (FromJSON, Value, eitherDecodeStrict, encode, object, parseJSON, toJSON, (.=))
import Data.Aeson.Types (parseEither)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Data.Foldable (toList)
import Invar.Use qualified as U
import Invar.Use.Statistics qualified as Statistics
import Numerical qualified
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)

run :: [String] -> IO ()
run ["--help"] = putStrLn usage
run [_, "--help"] = putStrLn usage
run ("inspect" : supplied) = do
    fields <- either die pure (O.parse options supplied)
    contract <- readContract fields
    observed <- observe fields contract
    Lazy.putStrLn (encode (U.describe observed))
run ("admit" : supplied) = do
    fields <- either die pure (O.parse options supplied)
    contract <- readContract fields
    observed <- observe fields contract
    let found = U.establish (U.Required (U.scope observed) (U.criterion contract)) observed
        decision = U.admit contract found
    Lazy.putStrLn
        ( encode
            ( object
                [ "contract" .= U.describeContract contract
                , "observation" .= U.describe observed
                , "finding" .= U.describeFinding found
                , "decision" .= U.describeDecision decision
                , "confidence" .= confidence contract observed
                ]
            )
        )
run ("statistics" : supplied) = do
    fields <- either die pure (O.parse options supplied)
    contract <- readContract fields
    observed <- observe fields contract
    requested <- maybe (die "Statistics need a loss requirement with a population standard") pure (U.lossRequirement (U.criterion contract))
    alpha <- case U.standard requested of
        U.HoeffdingPopulation population -> pure (U.regressionAlpha population)
        U.EmpiricalBernsteinPopulation population -> pure (U.regressionAlpha population)
        _ -> die "Statistics need a population standard"
    pairs <- maybe (die "Every unit needs a reference and a candidate loss") pure (traverse (\unit -> (,) <$> U.referenceLoss unit <*> U.candidateLoss unit) (toList (U.units observed)))
    result <- maybe (die "Statistics need at least two units and an alpha below one half") pure (Statistics.compare alpha (U.limit (U.regressionCeiling requested)) pairs)
    Lazy.putStrLn (encode (statistics alpha result))
run _ = die usage

readContract :: O.Fields -> IO U.UseContract
readContract fields = readInput fields "contract" >>= either die pure . U.decodeContract

observe :: O.Fields -> U.UseContract -> IO U.Observed
observe fields contract = do
    entries <- readInput fields "runs" >>= either die pure . eitherDecodeStrict
    cases <- traverse readCase entries
    either (die . show) pure (U.observe (U.BoundRun (U.declaredDomain contract) (U.declaredMeasurement contract) cases))
  where
    readCase :: [Value] -> IO U.Case
    readCase entry = case entry of
        [cohort, name, arguments] -> build cohort name arguments (toJSON ([] :: [[String]]))
        [cohort, name, arguments, repeated] -> build cohort name arguments repeated
        _ -> die "Each run entry is [cohort index, input key, paired-observation argument array] with an optional array of repeated candidate argument arrays"
    build cohort name arguments repeated = do
        key <- U.Key <$> decode cohort <*> decode name
        U.Case key <$> (decode arguments >>= Numerical.readPair) <*> (decode repeated >>= traverse Numerical.readRun)
    decode :: (FromJSON value) => Value -> IO value
    decode = either die pure . parseEither parseJSON

readInput :: O.Fields -> String -> IO Bytes.ByteString
readInput fields name = either die pure (O.required fields name) >>= Bytes.readFile

confidence :: U.UseContract -> U.Observed -> [Value]
confidence contract observed = case U.lossRequirement criterion of
    Just requested
        | Just selected <- claim (U.standard requested) ->
            [ object ["metric" .= show metric, "bound" .= U.describeConfidence bound]
            | (metric, budget) <- [(U.ReferenceLoss, U.referenceCeiling requested), (U.LossIncrease, U.regressionCeiling requested)]
            , Just bound <- [U.confidence (selected metric (U.limit budget)) observed]
            ]
    _ -> []
  where
    criterion = U.criterion contract
    claim (U.HoeffdingPopulation population) = Just (U.PopulationClaim (U.scope observed) population)
    claim (U.EmpiricalBernsteinPopulation population) = Just (U.EmpiricalBernsteinClaim (U.scope observed) population)
    claim _ = Nothing

statistics :: Rational -> Statistics.Statistics -> Value
statistics alpha result =
    object
        [ "units" .= Statistics.units result
        , "alpha" .= decimal alpha
        , "mean_increase" .= decimal (Statistics.meanIncrease result)
        , "hoeffding_upper" .= decimal (Statistics.hoeffdingUpper result)
        , "bernstein_upper" .= decimal (Statistics.bernsteinUpper result)
        , "wald" .= object ["lower" .= decimal (Statistics.waldLower result), "upper" .= decimal (Statistics.waldUpper result), "equivalent" .= Statistics.equivalent result, "noninferior" .= Statistics.noninferior result]
        , "mcnemar" .= fmap (\test -> object ["worse" .= Statistics.worse test, "better" .= Statistics.better test, "one_sided" .= decimal (Statistics.oneSided test)]) (Statistics.mcnemar result)
        ]
  where
    decimal value = fromRational value :: Double

options :: [OptDescr (String, String)]
options = O.descriptions [("contract", "Explicit domain, optional measurement and requirements in invar-use-contract JSON"), ("runs", "JSON array of [cohort index, input key, paired-observation argument array, optional array of repeated candidate argument arrays]")]

usage :: String
usage = usageInfo "Usage: invar use inspect --contract FILE --runs FILE\n       invar use admit --contract FILE --runs FILE\n       invar use statistics --contract FILE --runs FILE\nReads retained observations; does not execute models. Run arguments use compare numerical's input options, without --relation or --budget. Repeated candidate argument arrays use the unprefixed request, binding, log and exit-code options. Paths resolve from the current directory.\nExit zero means the judgment was computed; inspect decision.status for conditional admission, violation or unknown. Declared external reliance is retained, not authenticated." options
