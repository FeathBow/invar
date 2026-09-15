{-# LANGUAGE OverloadedStrings #-}

module CycleReplay (run) where

import Control.Monad (unless)
import Data.Aeson (Value, encode, object, (.=))
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8, encodeUtf8)
import HistoryInput qualified
import Invar.Learn qualified as Learn
import Invar.Policy qualified as Policy
import Invar.Replay.Cycle qualified as Cycle
import Invar.Replay.Native qualified as Native
import Invar.Replay.Update qualified as Update
import Invar.Store qualified as Store
import Invar.Workload qualified as Workload
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Directory (doesDirectoryExist)
import System.Exit (die)
import System.FilePath ((</>))
import Training qualified

run :: [String] -> IO ()
run [kind, "--help"] = putStrLn (usageInfo ("Usage: invar replay " ++ kind ++ " OPTIONS") (options kind))
run ("plan" : supplied) = do
    fields <- either die pure (O.parse (options "plan") supplied)
    observed <- reference fields
    either die emit (Cycle.describe observed)
run ("input" : supplied) = do
    fields <- either die pure (O.parse (options "input") supplied)
    settings <- either die pure (Training.settings fields)
    tasksPath <- either die pure (O.required fields "tasks")
    tasks <- Bytes.readFile tasksPath >>= either die pure . Workload.decode
    index <- either die pure (O.numeric fields "cohort")
    unless (index >= (0 :: Int)) (die "Expected a nonnegative cohort index")
    selected <- case drop index (Workload.cycles tasks) of
        workload : _ -> pure workload
        [] -> die "Selected cohort is outside the declared workload"
    call <- either die pure (O.numeric fields "call")
    path <- either die pure (O.required fields "log")
    expectedPath <- either die pure (O.required fields "expected")
    expected <- Bytes.readFile expectedPath >>= either die pure . Update.decodeBytes
    Bytes.readFile path >>= either die emit . Cycle.prepareInput (settings, expected) (selected, call)
run ("native-plan" : supplied) = do
    fields <- either die pure (O.parse (options "native-plan") supplied)
    (settings, tasks) <- native fields
    either die emit (Native.plan settings tasks)
run ("native-input" : supplied) = do
    fields <- either die pure (O.parse (options "native-input") supplied)
    (settings, tasks) <- native fields
    index <- either die pure (O.numeric fields "cohort")
    unless (index >= (0 :: Int)) (die "Expected a nonnegative cohort index")
    selected <- case drop index (Workload.cycles tasks) of
        workload : _ -> pure workload
        [] -> die "Selected cohort is outside the declared workload"
    path <- either die pure (O.required fields "log")
    Bytes.readFile path >>= either die emit . Native.input settings selected
run ("native-evaluate" : supplied) = do
    fields <- either die pure (O.parse (options "native-evaluate") supplied)
    (settings, tasks) <- native fields
    path <- either die pure (O.required fields "log")
    Bytes.readFile path >>= either die emit . Native.evaluate settings tasks
run ("native-quality" : supplied) = do
    fields <- either die pure (O.parse (options "native-quality") supplied)
    (settings, tasks) <- native fields
    trained <- either die pure (O.required fields "trained-policy")
    initialPath <- either die pure (O.required fields "initial-log")
    trainedPath <- either die pure (O.required fields "trained-log")
    initialLog <- Bytes.readFile initialPath
    trainedLog <- Bytes.readFile trainedPath
    either die emit (Native.quality (settings, trained) tasks (initialLog, trainedLog))
run ("publish" : supplied) = do
    fields <- either die pure (O.parse (options "publish") supplied)
    directory <- either die pure (O.required fields "output")
    staging <- either die pure (O.required fields "staging")
    destination <- either die pure (O.required fields "destination")
    method <- either die pure (O.required fields "publication")
    selected <- case method of
        "reference" -> pure Store.LinkImmutable
        "rename" -> pure Store.RenameExclusive
        _ -> die "Expected rename or reference publication"
    let target = Store.Location directory (encodeUtf8 (Text.pack staging)) (encodeUtf8 (Text.pack destination))
    Store.validateLocation target
    description <- Bytes.getContents >>= either die pure . Policy.decodeDescription
    actual <- Policy.identity (directory </> staging </> "adapter.safetensors")
    unless (actual == Policy.adapter description) (die "Staged adapter differs from the publication policy description")
    Policy.stageDescription (directory </> staging </> "policy.json") description
    receipt <- Store.publishCheckpoint selected target
    let location = Store.location receipt
    emit (object ["output" .= Store.directory location, "staging" .= decodeUtf8 (Store.staging location), "destination" .= decodeUtf8 (Store.destination location), "publication" .= method, "scope" .= ("actual durable filesystem publication; no numerical or invocation evidence" :: String)])
run ("inspect" : supplied) = do
    fields <- either die pure (O.parse (options "inspect") supplied)
    expected <- reference fields
    directory <- either die pure (O.required fields "replay-output")
    status <- either die pure (O.numeric fields "replay-exit-code")
    path <- either die pure (O.required fields "replay-log")
    encoded <- Bytes.readFile path
    Cycle.inspect expected (directory, status, encoded) >>= emit
run _ = die "Usage: invar replay {plan|input|publish|inspect|native-plan|native-input|native-evaluate|native-quality} OPTIONS"

native :: O.Fields -> IO (Learn.Settings, Workload.Document)
native fields = do
    settings <- either die pure (Training.settings fields)
    path <- either die pure (O.required fields "tasks")
    tasks <- Bytes.readFile path >>= either die pure . Workload.decode
    pure (settings, tasks)

reference :: O.Fields -> IO Cycle.Reference
reference fields = do
    initial <- either die pure (O.required fields "initial")
    (run_, tasks, encoded) <- HistoryInput.trace fields
    Cycle.admit doesDirectoryExist (initial, run_) (tasks, encoded)

options :: String -> [OptDescr (String, String)]
options "plan" = HistoryInput.traceOptions ++ O.descriptions [("initial", "Reference initial checkpoint directory")]
options "input" = Training.settingsOptions ++ O.descriptions [("tasks", "Frozen workload"), ("cohort", "Zero-based cycle index"), ("call", "Update call index"), ("log", "Actual completed inference groups"), ("expected", "Core-admitted reference update document")]
options "native-plan" = Training.settingsOptions ++ O.descriptions [("tasks", "Frozen workload")]
options "native-input" = options "native-plan" ++ O.descriptions [("cohort", "Zero-based cycle index"), ("log", "Actual named numerical results, one JSON object per line")]
options "native-evaluate" = options "native-plan" ++ O.descriptions [("log", "Named numerical results for one policy in cohort order")]
options "native-quality" = options "native-plan" ++ O.descriptions [("trained-policy", "Trained policy identity under the same native model profile"), ("initial-log", "Initial named numerical results in cohort order"), ("trained-log", "Trained named numerical results in cohort order")]
options "publish" = O.descriptions [("output", "Existing publication directory; complete successor policy description is read from stdin"), ("staging", "Staged checkpoint entry"), ("destination", "Absent publication entry"), ("publication", "rename or reference")]
options "inspect" = options "plan" ++ O.descriptions [("replay-output", "Actual replay publication directory"), ("replay-log", "Complete actual replay transcript"), ("replay-exit-code", "Independently observed replay exit status")]
options _ = []

emit :: Value -> IO ()
emit = Lazy.putStrLn . encode
