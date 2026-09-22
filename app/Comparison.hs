{-# LANGUAGE OverloadedStrings #-}

module Comparison (run, inspect) where

import Control.Monad (unless)
import Data.Aeson (Value, encode, object, (.=))
import Data.ByteString.Lazy.Char8 qualified as Lazy
import HistoryInput qualified
import Invar.Learn qualified as Learn
import Invar.Learn.Observation qualified as Observation
import Invar.Learn.Report qualified as Report
import Invar.Learn.State qualified as State
import NativeCodec qualified
import Numerical qualified
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)
import Training qualified

run :: [String] -> IO ()
run ["--help"] = putStrLn usage
run ("numerical" : supplied) = Numerical.run supplied
run ("histories" : supplied) = HistoryInput.compareHistories supplied
run ["initial", "--help"] = putStrLn (usageInfo "Usage: invar compare initial OPTIONS" initialOptions)
run ("initial" : supplied) = do
    fields <- either die pure (O.parse initialOptions supplied)
    settings <- either die pure (Training.settings fields)
    left <- either die pure (O.required fields "left-checkpoint")
    right <- either die pure (O.required fields "right-checkpoint")
    rightPolicy <- either die pure (O.required fields "right-policy")
    rightLearner <- either die pure (O.required fields "right-learner")
    codec <- either die pure (NativeCodec.select fields)
    State.compareInitial codec (settings, left) (settings {Learn.policy = rightPolicy, Learn.learner = rightLearner}, right) >>= emit
run [kind, "--help"] | kind `elem` ["gradients", "probabilities", "states"] = putStrLn usage
run (kind : supplied) | kind `elem` ["gradients", "probabilities", "states"] = do
    let artifact = if kind == "states" then "checkpoint" else kind
        options = pairOptions artifact ++ comparisonOptions kind
    fields <- either die pure (O.parse options supplied)
    left <- either die pure (input fields "left-" artifact)
    right <- either die pure (input fields "right-" artifact)
    case kind of
        "gradients" -> do
            policy <- either die pure (O.required fields "policy")
            Observation.gradients policy left right >>= emit
        "states" -> do
            policy <- either die pure (O.required fields "policy")
            codec <- either die pure (NativeCodec.select fields)
            State.compare codec policy (left, right) >>= emit
        _ -> Observation.probabilities left right >>= emit
run _ = die usage

comparisonOptions :: String -> [OptDescr (String, String)]
comparisonOptions kind = O.descriptions [("policy", "Consumed adapter checkpoint") | kind /= "probabilities"] ++ if kind == "states" then NativeCodec.options else []

initialOptions :: [OptDescr (String, String)]
initialOptions = Training.settingsOptions ++ O.descriptions [("left-checkpoint", "Initial checkpoint directory matching --policy and --learner"), ("right-checkpoint", "Other complete initial checkpoint directory"), ("right-policy", "Other initial canonical adapter identity"), ("right-learner", "Other initial learner file identity")] ++ NativeCodec.options

inspect :: String -> [String] -> IO ()
inspect _ ["--help"] = putStrLn usage
inspect "updates" supplied = do
    fields <- either die pure (O.parse (pairOptions "") supplied)
    left <- either die pure (input fields "left-" "") >>= Observation.report
    right <- either die pure (input fields "right-" "") >>= Observation.report
    either die pure (Report.paired left right)
    emit (object ["left" .= Report.describe left, "right" .= Report.describe right])
inspect kind supplied = do
    let artifact = if kind == "probabilities" then kind else ""
    fields <- either die pure (O.parse (inputOptions "" artifact ++ O.descriptions [("log-digest", "Expected prior log snapshot identity")]) supplied)
    source <- either die pure (input fields "" artifact)
    expected <- Observation.report source
    case O.optional fields "log-digest" of
        Just digest -> unless (digest == Report.logDigest expected) (die "Update log identity changed after inspection")
        Nothing -> pure ()
    case kind of
        "update" -> emit (Report.describe expected)
        "probabilities" -> Observation.probability expected (Observation.artifact source) >>= emit . object . pure . ("samples" .=)
        _ -> die usage

input :: O.Fields -> String -> String -> Either String Observation.Input
input fields prefix kind = do
    logPath <- O.required fields (prefix ++ "log")
    call <- O.numeric fields (prefix ++ "call")
    artifact <- if null kind then pure "" else O.required fields (prefix ++ kind)
    pure Observation.Input {Observation.logPath, Observation.call, Observation.artifact}

inputOptions :: String -> String -> [OptDescr (String, String)]
inputOptions prefix kind = O.descriptions ([(prefix ++ "log", "Execution log snapshot"), (prefix ++ "call", "Selected update call")] ++ [(prefix ++ kind, "Bound artifact snapshot") | not (null kind)])

pairOptions :: String -> [OptDescr (String, String)]
pairOptions kind = concatMap (`inputOptions` kind) ["left-", "right-"]

emit :: Value -> IO ()
emit = Lazy.putStrLn . encode

usage :: String
usage = usageInfo "Usage: invar compare gradients|probabilities|states|initial|histories OPTIONS\n       invar compare numerical OPTIONS\n       invar inspect update|updates|probabilities OPTIONS\nCompare conditional, bound artifact observations. Numerical comparisons emit finite accept/refute/unknown findings; see compare numerical --help." (pairOptions "gradients" ++ pairOptions "probabilities" ++ pairOptions "checkpoint" ++ comparisonOptions "states")
