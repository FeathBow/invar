{-# LANGUAGE OverloadedStrings #-}

module Use (run) where

import Data.Aeson (Value, encode, object, (.=))
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Data.Foldable (toList)
import Invar.Use qualified as U
import Invar.Use.Statistics qualified as Statistics
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)
import UsePlan qualified
import UseRuns qualified

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
                , "bounds" .= [object ["metric" .= show metric, "bound" .= U.describeConfidence bound] | (metric, bound) <- U.bounds found]
                , "statistics" .= statistics contract observed
                ]
            )
        )
run ("plan" : supplied) = UsePlan.run supplied
run _ = die usage

readContract :: O.Fields -> IO U.UseContract
readContract fields = readInput fields "contract" >>= either die pure . U.decodeContract

observe :: O.Fields -> U.UseContract -> IO U.Observed
observe fields contract = either die pure (O.required fields "runs") >>= (`UseRuns.observe` contract) >>= either die pure

readInput :: O.Fields -> String -> IO Bytes.ByteString
readInput fields name = either die pure (O.required fields name) >>= Bytes.readFile

statistics :: U.UseContract -> U.Observed -> Maybe Value
statistics contract observed = do
    (population, budget, used) <- case [claim | claim <- U.lossClaims (U.scope observed) (U.criterion contract), increase claim] of
        [U.PopulationClaim _ population _ budget] -> Just (population, budget, Statistics.Hoeffding)
        [U.EmpiricalBernsteinClaim _ population _ budget] -> Just (population, budget, Statistics.EmpiricalBernstein)
        _ -> Nothing
    pairs <- traverse (\unit -> (,) <$> U.referenceLoss unit <*> U.candidateLoss unit) (toList (U.units observed))
    result <- Statistics.compare (U.regressionAlpha population) budget used pairs
    let (other, upper) = Statistics.alternative result
    pure
        ( object
            [ "units" .= Statistics.units result
            , "mean_increase" .= decimal (Statistics.meanIncrease result)
            , "alternative_bound" .= object ["method" .= show other, "upper" .= decimal upper]
            , "wald" .= object ["lower" .= decimal (Statistics.waldLower result), "upper" .= decimal (Statistics.waldUpper result), "equivalent" .= Statistics.equivalent result, "noninferior" .= Statistics.noninferior result]
            , "mcnemar" .= fmap (\test -> object ["worse" .= Statistics.worse test, "better" .= Statistics.better test, "one_sided" .= decimal (Statistics.oneSided test)]) (Statistics.mcnemar result)
            ]
        )
  where
    increase claim = case claim of
        U.PopulationClaim _ _ U.LossIncrease _ -> True
        U.EmpiricalBernsteinClaim _ _ U.LossIncrease _ -> True
        _ -> False
    decimal value = fromRational value :: Double

options :: [OptDescr (String, String)]
options = O.descriptions [("contract", "Explicit domain, optional measurement and requirements in invar-use-contract JSON"), ("runs", "JSON array of [cohort index, input key, paired-observation argument array, optional array of repeated candidate argument arrays]")]

usage :: String
usage = usageInfo "Usage: invar use inspect --contract FILE --runs FILE\n       invar use admit --contract FILE --runs FILE\nReads retained observations; does not execute models. Run arguments use compare numerical's input options, without --relation or --budget. Repeated candidate argument arrays use the unprefixed request, binding, log and exit-code options. Paths resolve from the current directory.\nExit zero means the judgment was computed; inspect decision.status for conditional admission, violation or unknown. Declared external reliance is retained, not authenticated." options
