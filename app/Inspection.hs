{-# LANGUAGE OverloadedStrings #-}

module Inspection (run) where

import Comparison qualified
import Control.Monad (unless, when)
import Data.Aeson (Value, encode, object, toJSON, (.=))
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Data.Maybe (isJust)
import EvaluationInput qualified
import HistoryInput qualified
import InferenceInput qualified
import Invar.Evaluation qualified as Evaluation
import Invar.History.Trace qualified as Trace
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Observation qualified as Observation
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Session qualified as Session
import Invar.Workload qualified as Workload
import Options qualified as O
import System.Console.GetOpt (OptDescr (Option), usageInfo)
import System.Exit (die)

run :: [String] -> IO ()
run ["--help"] = putStrLn usage
run ["inference", "--help"] = putStrLn (usageInfo "Usage: invar inspect inference OPTIONS" inferenceOptions)
run ["trace", "--help"] = putStrLn (usageInfo "Usage: invar inspect trace OPTIONS" HistoryInput.traceOptions)
run ("history" : supplied) = HistoryInput.inspect supplied
run ("initial" : supplied) = HistoryInput.inspectInitial supplied
run ("trace" : supplied) = do
    fields <- either die pure (O.parse HistoryInput.traceOptions supplied)
    (declaredRun, declared, encoded) <- HistoryInput.trace fields
    initial <- HistoryInput.initialPolicy fields
    observed <- either die pure (Trace.admit declaredRun initial declared encoded)
    emit (object ["tasks_sha256" .= Workload.digest declared, "observation" .= Trace.describe observed])
run ("inference" : supplied) = do
    fields <- either die pure (O.parse inferenceOptions supplied)
    path <- either die pure (O.required fields "log")
    status <- either die pure (O.numeric fields "exit-code")
    encoded <- Bytes.readFile path
    (protocol, declared) <- case O.optional fields "calls" of
        Just batch -> do
            when (any (isJust . O.optional fields) [name | Option _ names _ _ <- InferenceInput.declarationWith "", name <- names]) (die "--calls declares every member of the batch; request and binding options are invalid")
            (Session.Batched,) <$> InferenceInput.readCalls batch
        Nothing -> do
            planned <- InferenceInput.declaredWith "" fields
            bound <- either die pure (InferenceInput.binding fields)
            (Session.Single,) . pure <$> either (die . show) pure (Call.prepare bound planned)
    admitted <- either (die . show) pure (Replay.standalone protocol (Session.Declaration declared Nothing) (Replay.declared status) encoded)
    let observed = map Observation.view admitted
    case O.optional fields "log-digest" of
        Nothing -> pure ()
        Just digest -> unless (all ((== digest) . Observation.logDigest) observed) (die "Inference log identity changed after inspection")
    case (protocol, observed) of
        (Session.Single, [single]) -> emit (Observation.describe single)
        _ -> emit (toJSON (map Observation.describe observed))
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
    selected <- EvaluationInput.declared "" fields
    encoded <- Bytes.readFile reportPath
    either die (emit . Evaluation.describe) (Evaluation.admit expected selected encoded)
run _ = die usage

readTasks :: FilePath -> IO Workload.Document
readTasks path = Bytes.readFile path >>= either die pure . Workload.decode

emit :: Value -> IO ()
emit = Lazy.putStrLn . encode

taskOptions :: [OptDescr (String, String)]
taskOptions = O.descriptions [("input", "Frozen workload file")]

evaluationOptions :: [OptDescr (String, String)]
evaluationOptions = O.descriptions [("tasks", "Frozen workload file"), ("tasks-digest", "Expected identity from a prior task inspection, when supplied")] ++ EvaluationInput.options ""

inferenceOptions :: [OptDescr (String, String)]
inferenceOptions = InferenceInput.declarationWith "" ++ O.descriptions [("calls", "Calls array given to invar infer batch, declaring every member of a batch log in place of the request options"), ("log", "Complete standalone inference stdout"), ("exit-code", "Independently observed inference process exit status"), ("log-digest", "Expected prior log snapshot identity")]

usage :: String
usage = usageInfo "Usage: invar inspect tasks --input FILE\n       invar inspect evaluation OPTIONS\n       invar inspect inference OPTIONS\n       invar inspect trace OPTIONS\n       invar inspect history OPTIONS\n       invar inspect initial OPTIONS\nValidate complete input snapshots and emit their checked observations." evaluationOptions
