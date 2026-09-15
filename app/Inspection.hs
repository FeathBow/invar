{-# LANGUAGE OverloadedStrings #-}

module Inspection (run) where

import Comparison qualified
import Control.Monad (unless)
import Data.Aeson (Value, encode, object, (.=))
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import HistoryInput qualified
import InferenceInput qualified
import Invar.Evaluation qualified as Evaluation
import Invar.History.Cohort qualified as Cohort
import Invar.History.Trace qualified as Trace
import Invar.Infer qualified as Infer
import Invar.Infer.Observation qualified as Observation
import Invar.Workload qualified as Workload
import Options qualified as O
import ReplayInput qualified
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)
import Training qualified
import UpdateReplayInput qualified

run :: [String] -> IO ()
run ["--help"] = putStrLn usage
run ["inference", "--help"] = putStrLn (usageInfo "Usage: invar inspect inference OPTIONS" inferenceOptions)
run ["cohort", "--help"] = putStrLn (usageInfo "Usage: invar inspect cohort OPTIONS" cohortOptions)
run ["trace", "--help"] = putStrLn (usageInfo "Usage: invar inspect trace OPTIONS" HistoryInput.traceOptions)
run ("history" : supplied) = HistoryInput.inspect supplied
run ("initial" : supplied) = HistoryInput.inspectInitial supplied
run (kind : supplied) | kind `elem` ["update-calls", "update-output"] = UpdateReplayInput.run kind supplied
run (kind : supplied) | kind `elem` ["replay-calls", "replay-output", "measurements"] = ReplayInput.run kind supplied
run ("trace" : supplied) = do
    fields <- either die pure (O.parse HistoryInput.traceOptions supplied)
    (declaredRun, declared, encoded) <- HistoryInput.trace fields
    observed <- either die pure (Trace.admit declaredRun declared encoded)
    emit (object ["tasks_sha256" .= Workload.digest declared, "observation" .= Trace.describe observed])
run ("cohort" : supplied) = do
    fields <- either die pure (O.parse cohortOptions supplied)
    settings <- either die pure (Training.settings fields)
    path <- either die pure (O.required fields "tasks")
    declared <- readTasks path
    index <- either die pure (O.numeric fields "cohort")
    unless (index >= (0 :: Int)) (die "Expected a nonnegative cohort index")
    selected <- case drop index (Workload.cycles declared) of
        workload : _ -> pure workload
        [] -> die "Selected cohort is outside the declared workload"
    call <- either die pure (O.numeric fields "call")
    logPath <- either die pure (O.required fields "log")
    observed <- Bytes.readFile logPath >>= either die pure . Cohort.admitLog settings (selected, call)
    emit (object ["tasks_sha256" .= Workload.digest declared, "cohort" .= index, "observation" .= Cohort.describe observed])
run ("inference" : supplied) = do
    fields <- either die pure (O.parse inferenceOptions supplied)
    requested <- either die pure (InferenceInput.request fields)
    planned <- either (die . show) pure (Infer.prepare requested)
    bound <- either die pure (InferenceInput.binding fields)
    path <- either die pure (O.required fields "log")
    status <- either die pure (O.numeric fields "exit-code")
    unless (status == (0 :: Int)) (die "Inference process did not exit successfully")
    observed <- Bytes.readFile path >>= either die pure . Observation.admit planned bound
    case O.optional fields "log-digest" of
        Nothing -> pure ()
        Just digest -> unless (digest == Observation.logDigest observed) (die "Inference log identity changed after inspection")
    emit (Observation.describe observed)
run (kind : supplied) | kind `elem` ["update", "updates", "probabilities"] = Comparison.inspect kind supplied
run ("tasks" : supplied) = do
    fields <- either die pure (O.parse taskOptions supplied)
    path <- either die pure (O.required fields "input")
    expected <- readTasks path
    emit (Workload.describe expected)
run ("evaluation" : supplied) = do
    fields <- either die pure (O.parse evaluationOptions supplied)
    path <- either die pure (O.required fields "tasks")
    expected <- readTasks path
    case O.optional fields "tasks-digest" of
        Nothing -> pure ()
        Just digest -> unless (digest == Workload.digest expected) (die "Workload input identity changed after inspection")
    reportPath <- either die pure (O.required fields "log")
    policy <- either die pure (O.required fields "policy")
    status <- either (die . ("Evaluation process: " ++)) pure (O.numeric fields "exit-code")
    encoded <- Bytes.readFile reportPath
    either die (emit . Evaluation.describe) (Evaluation.admit expected (Evaluation.Run policy status) encoded)
run _ = die usage

readTasks :: FilePath -> IO Workload.Document
readTasks path = Bytes.readFile path >>= either die pure . Workload.decode

emit :: Value -> IO ()
emit = Lazy.putStrLn . encode

taskOptions :: [OptDescr (String, String)]
taskOptions = O.descriptions [("input", "Frozen workload file")]

evaluationOptions :: [OptDescr (String, String)]
evaluationOptions = O.descriptions [("tasks", "Frozen workload file"), ("tasks-digest", "Expected identity from a prior task inspection, when supplied"), ("log", "Complete invar evaluate stdout"), ("policy", "Expected canonical adapter identity"), ("exit-code", "Independently observed evaluation process exit status")]

inferenceOptions :: [OptDescr (String, String)]
inferenceOptions = InferenceInput.options ++ O.descriptions [("log", "Complete standalone inference stdout"), ("exit-code", "Independently observed inference process exit status"), ("log-digest", "Expected prior log snapshot identity")]

cohortOptions :: [OptDescr (String, String)]
cohortOptions = Training.settingsOptions ++ O.descriptions [("tasks", "Frozen workload file"), ("cohort", "Zero-based declared cohort index"), ("call", "Selected update call"), ("log", "Training log snapshot")]

usage :: String
usage = usageInfo "Usage: invar inspect tasks --input FILE\n       invar inspect evaluation OPTIONS\n       invar inspect inference OPTIONS\n       invar inspect cohort OPTIONS\n       invar inspect trace OPTIONS\n       invar inspect history OPTIONS\n       invar inspect initial OPTIONS\n       invar inspect replay-calls OPTIONS\n       invar inspect replay-output OPTIONS\n       invar inspect measurements OPTIONS\n       invar inspect update-calls OPTIONS\n       invar inspect update-output OPTIONS\nValidate complete input snapshots and emit their checked observations." evaluationOptions
