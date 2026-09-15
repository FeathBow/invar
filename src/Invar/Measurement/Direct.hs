{-# LANGUAGE OverloadedStrings #-}

module Invar.Measurement.Direct (admit) where

import Control.Monad (unless)
import Data.Aeson (Object, Value (..), withObject, (.:))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString.Char8 qualified as Bytes
import Invar.Evaluation qualified as Evaluation
import Invar.Json qualified as Json
import Invar.Measurement.Direct.Record (checkStderr, document, padded, positive, records)
import Invar.Measurement.Direct.Resident qualified as Resident
import Invar.Measurement.Duration qualified as Duration
import Invar.Measurement.Run qualified as Run
import Invar.Measurement.Source qualified as Source
import Invar.Measurement.Stream qualified as Stream
import Invar.Replay.Call (cohorts)
import Invar.Replay.Call qualified as Call
import Invar.Replay.Inference qualified as Replay
import Numeric.Natural (Natural)
import System.FilePath ((</>))

data Complete = Complete {mode :: Replay.Mode, wall :: Double, process :: Double, responseTokens :: Natural, equalResults :: Natural, loadCount :: Natural}
data Observed = Observed {cohort :: Natural, session :: Stream.Session, processSeconds :: Double, equal :: Natural}

admit :: Source.Source -> FilePath -> Replay.Reference -> IO Run.Run
admit source root reference = do
    (completeSnapshot, supplied) <- document source (root </> "complete.json")
    if Fields.lookup "mode" supplied == Just (String "resident")
        then Resident.admit source (root, completeSnapshot) (reference, supplied)
        else finite source (root, completeSnapshot) (reference, supplied)

finite :: Source.Source -> (FilePath, Source.Snapshot) -> (Replay.Reference, Object) -> IO Run.Run
finite source (root, completeSnapshot) (reference, supplied) = do
    let expected = Replay.calls reference
        grouped = cohorts expected
    finished <- Source.checked (parseEither (complete reference) supplied)
    callsSnapshot <- Source.snapshot source (root </> "calls.jsonl")
    rows <- Source.checked (records (Source.encoded callsSnapshot))
    unless (length rows == length expected) (Source.invalid "Incomplete direct measurements")
    observed <- case mode finished of
        Replay.Process -> traverse (processRecord source root) (zip3 [0 ..] expected rows)
        selected -> traverse (sessionRecord source (root, selected)) (partition grouped rows)
    duration <-
        Source.checked
            ( case mode finished of
                Replay.Process -> Duration.total (map processSeconds observed)
                _ -> Duration.checkedSeconds (foldl' (\total item -> total + processSeconds item) 0 observed)
            )
    let loaded = map session observed
        tokens = sum [Call.responseTokens (Stream.call sample) | item <- loaded, sample <- Stream.samples item]
        equalCount = sum (map equal observed)
    unless (duration == process finished && responseTokens finished == tokens && equalResults finished == equalCount && loadCount finished == fromIntegral (length loaded)) (Source.invalid "Direct completion differs from its call records")
    unless (wall finished >= duration) (Source.invalid "Campaign duration is shorter than its subprocesses")
    critical <- Source.checked (Duration.total (map Stream.elapsed loaded))
    pure
        Run.Run
            { Run.logDigest = Source.digest callsSnapshot
            , Run.schedule = Run.Finite [(index, [session item | item <- observed, cohort item == index]) | (index, _) <- grouped]
            , Run.concurrent = False
            , Run.criticalPath = critical
            , Run.equalResults = equalCount
            , Run.completion = Just (Run.Completion (Source.digest completeSnapshot) (wall finished) duration [])
            }

complete :: Replay.Reference -> Object -> Parser Complete
complete reference fields = do
    let declared = any (`Fields.member` fields) ["mode", "cohorts", "loads"]
        expected = Replay.calls reference
        evaluated = Replay.evaluation reference
    Json.fields (["reference_log_sha256", "tasks_sha256", "policy", "calls", "wall_seconds", "process_seconds", "response_tokens", "equal_results", "scope"] ++ [key | declared, key <- ["mode", "cohorts", "loads"]]) fields
    referenceDigest <- fields .: "reference_log_sha256"
    tasksDigest <- fields .: "tasks_sha256"
    policy <- fields .: "policy"
    unless (referenceDigest == Evaluation.logDigest evaluated && tasksDigest == Evaluation.inputDigest evaluated && policy == Evaluation.policy evaluated) (fail "Direct completion belongs to a different reference")
    calls <- fields .: "calls" :: Parser Natural
    unless (calls == fromIntegral (length expected)) (fail "Incomplete direct measurements")
    selected <-
        if declared
            then do
                declaredMode <- fields .: "mode" :: Parser String
                count <- fields .: "cohorts" :: Parser Natural
                unless (count == fromIntegral (length (cohorts expected))) (fail "Direct replay cohort count differs from the reference")
                case declaredMode of
                    "process" -> pure Replay.Process
                    "session" -> pure Replay.Session
                    "batch" -> pure Replay.Batched
                    _ -> fail "Unknown direct replay mode"
            else pure Replay.Process
    loads <- if declared then fields .: "loads" else pure calls
    Complete selected <$> positive "wall_seconds" fields <*> positive "process_seconds" fields <*> fields .: "response_tokens" <*> fields .: "equal_results" <*> pure loads

processRecord :: Source.Source -> FilePath -> (Natural, Call.Call, Object) -> IO Observed
processRecord source root (index, expected, reported) = do
    duration <- Source.checked (parseEither validate reported)
    let prefix = root </> padded index
    (_, status) <- document source (prefix ++ ".status.json")
    Source.checked (parseEither (Json.fields ["index", "binding", "exit_code", "process_seconds", "stderr_sha256"]) status)
    unless (all (\(key, value) -> Fields.lookup key reported == Just value) (Fields.toList status)) (Source.invalid "Direct status differs from the measurement record")
    checkStderr source (prefix ++ ".stderr.log") reported
    raw <- Source.snapshot source (prefix ++ ".stdout.jsonl")
    actual <- Source.checked (Replay.observe (Replay.Process, 0) [expected] (Source.encoded raw) >>= parseEither (withObject "direct process observation" pure))
    unless (all (\(key, value) -> Fields.lookup key reported == Just value) (Fields.toList actual)) (Source.invalid "Direct measurement differs from raw output")
    measured <- oneSession (Call.cohort expected, 1) (Source.encoded raw)
    same <- Source.checked (parseEither (.: "result_equal") actual)
    pure (Observed (Call.cohort expected) measured duration (if same then 1 else 0))
  where
    validate fields = do
        Json.fields ["index", "binding", "exit_code", "process_seconds", "stderr_sha256", "stdout_sha256", "result_equal", "response_tokens", "load", "inference"] fields
        actualIndex <- fields .: "index"
        status <- fields .: "exit_code" :: Parser Int
        unless (actualIndex == index && status == 0) (fail "Failed or reordered direct process")
        unless (Fields.lookup "binding" fields == Fields.lookup "binding" (Call.consumed expected)) (fail "Direct measurement binding mismatch")
        positive "process_seconds" fields

sessionRecord :: Source.Source -> (FilePath, Replay.Mode) -> ((Natural, [Call.Call]), [Object]) -> IO Observed
sessionRecord source (root, mode) ((index, expected), reported) = do
    let prefix = root </> ("session-" ++ padded index)
    (_, status) <- document source (prefix ++ ".status.json")
    duration <- Source.checked (parseEither validate status)
    checkStderr source (prefix ++ ".stderr.log") status
    raw <- Source.snapshot source (prefix ++ ".stdout.jsonl")
    actual <- Source.checked (Replay.observe (mode, 0) expected (Source.encoded raw) >>= parseEither (withObject "direct session observation" (.: "calls")))
    let withCohort = Fields.insert "cohort" (Number (fromIntegral index))
    unless (reported == map withCohort actual) (Source.invalid "Session call record differs from raw output")
    measured <- oneSession (index, length expected) (Source.encoded raw)
    equalCount <- Source.checked (traverse (parseEither (.: "result_equal")) actual)
    pure (Observed index measured duration (fromIntegral (length (filter id equalCount))))
  where
    validate fields = do
        Json.fields ["cohort", "exit_code", "process_seconds", "calls", "stderr_sha256"] fields
        actualIndex <- fields .: "cohort"
        status <- fields .: "exit_code" :: Parser Int
        count <- fields .: "calls" :: Parser Natural
        unless (actualIndex == index && status == 0 && count == fromIntegral (length expected)) (fail "Failed or incomplete session replay")
        positive "process_seconds" fields

oneSession :: (Natural, Int) -> Bytes.ByteString -> IO Stream.Session
oneSession (index, count) encoded = do
    measured <- Source.checked (Stream.admit index encoded)
    case measured of
        [single] | length (Stream.samples single) == count -> pure single
        _ -> Source.invalid "Expected one model load and one result per call in a direct replay"

partition :: [(Natural, [Call.Call])] -> [Object] -> [((Natural, [Call.Call]), [Object])]
partition [] _ = []
partition (group@(_, members) : rest) rows = let (first, remaining) = splitAt (length members) rows in (group, first) : partition rest remaining
