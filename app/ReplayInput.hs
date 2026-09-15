module ReplayInput (run) where

import Control.Monad (when)
import Data.Aeson (encode)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Data.Maybe (isJust)
import Invar.Evaluation qualified as Evaluation
import Invar.Measurement.Inference qualified as Measurement
import Invar.Replay.Inference qualified as Replay
import Invar.Workload qualified as Workload
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)

run :: String -> [String] -> IO ()
run kind ["--help"] = putStrLn (usageInfo ("Usage: invar inspect " ++ kind ++ " OPTIONS") (options kind))
run kind supplied = do
    fields <- either die pure (O.parse (options kind) supplied)
    selectedMode <- if kind == "measurements" then pure Replay.Process else either die pure (mode fields)
    when (selectedMode /= Replay.Resident && isJust (O.optional fields "owner")) (die "--owner requires resident replay output")
    logPath <- either die pure (O.required fields "log")
    encoded <- Bytes.readFile logPath
    status <- either die pure (O.numeric fields "exit-code")
    observed <- case kind of
        "replay-calls" -> do
            taskPath <- either die pure (O.required fields "tasks")
            tasks <- Bytes.readFile taskPath >>= either die pure . Workload.decode
            policy <- either die pure (O.required fields "policy")
            either die (pure . encode . Replay.describe) (Replay.admit tasks (Evaluation.Run policy status, selectedMode) encoded)
        "replay-output" ->
            if selectedMode == Replay.Resident
                then do
                    owner <- either die pure (O.numeric fields "owner")
                    expected <- Bytes.getContents >>= either die pure . Replay.decodeResidentCalls
                    either die (pure . encode) (Replay.observeResident (owner, status) expected encoded)
                else do
                    expected <- Bytes.getContents >>= either die pure . Replay.decodeCalls
                    either die (pure . encode) (Replay.observe (selectedMode, status) expected encoded)
        "measurements" -> do
            taskPath <- either die pure (O.required fields "tasks")
            tasks <- Bytes.readFile taskPath >>= either die pure . Workload.decode
            policy <- either die pure (O.required fields "policy")
            either die (pure . encode) (Measurement.admit tasks (Evaluation.Run policy status) encoded)
        _ -> die "Unknown inference replay inspection"
    Lazy.putStrLn observed

mode :: O.Fields -> Either String Replay.Mode
mode fields = case O.optional fields "mode" of
    Just "process" -> Right Replay.Process
    Just "session" -> Right Replay.Session
    Just "batch" -> Right Replay.Batched
    Just "resident" -> Right Replay.Resident
    _ -> Left "Expected explicit replay --mode process, session, batch or resident"

options :: String -> [OptDescr (String, String)]
options kind = O.descriptions ([("log", "Complete reference or direct worker stdout"), ("exit-code", "Independently observed process exit status")] ++ [("mode", "process, session, batch or resident") | kind /= "measurements"] ++ [("owner", "Physical owner index required for resident replay output") | kind == "replay-output"] ++ reference)
  where
    reference = if kind `elem` ["replay-calls", "measurements"] then [("tasks", "Frozen reference workload"), ("policy", "Expected reference policy identity")] else []
