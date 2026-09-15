{-# LANGUAGE OverloadedStrings #-}

module Invar.Measurement.Direct.Resident (admit) where

import Control.Monad (unless)
import Data.Aeson (Object, Value (..), object, withObject, (.:), (.=))
import Data.Aeson.Types (Parser, parseEither)
import Data.Text.Encoding (decodeUtf8)
import Invar.Evaluation qualified as Evaluation
import Invar.Json qualified as Json
import Invar.Measurement.Direct.Record (checkStderr, document, padded, positive, records)
import Invar.Measurement.Direct.Timing qualified as Timing
import Invar.Measurement.Resident qualified as Measurement
import Invar.Measurement.Run qualified as Run
import Invar.Measurement.Source qualified as Source
import Invar.Replay.Call qualified as Call
import Invar.Replay.Inference qualified as Replay
import Invar.Replay.Resident qualified as Resident
import Invar.Resident qualified as Boundary
import Invar.Resident.Inference qualified as Physical
import Numeric.Natural (Natural)
import System.FilePath ((</>))

data Complete = Complete {wall :: Double, process :: Double, responseTokens :: Natural, equalResults :: Natural, loads :: Natural, intervals :: ([Timing.Interval], [Timing.Interval])}
data Observed = Observed {timing :: Timing.Status, observedOutput :: Resident.Output, provenance :: Value}

admit :: Source.Source -> (FilePath, Source.Snapshot) -> (Replay.Reference, Object) -> IO Run.Run
admit source (root, snapshot) (reference, fields) = do
    owners <- maybe (Source.invalid "Resident direct replay requires a resident reference") pure (Replay.ownerCalls reference)
    finished <- Source.checked (parseEither (complete (reference, owners)) fields)
    observed <- traverse (owner source root) owners
    Source.checked (Timing.validate (wall finished, process finished) (intervals finished) (map timing observed))
    let outputs = map observedOutput observed
        grouped = [(index, [group | output <- outputs, (cohort, group) <- Resident.groups output, cohort == index]) | (index, _) <- Call.cohorts (Replay.calls reference)]
        closed = [(Boundary.Owner Boundary.Inference (Resident.owner output), Resident.closed output, Resident.closing output) | output <- reverse outputs]
    measured <- Source.checked (Measurement.admit (Physical.Ledger (fromIntegral (length owners)) grouped closed))
    callSnapshot <- Source.snapshot source (root </> "calls.jsonl")
    actualRows <- Source.checked (records (Source.encoded callSnapshot))
    expectedRows <- Source.checked (traverse (parseEither (withObject "direct resident call" pure)) [row | (index, _) <- grouped, output <- outputs, row <- Resident.rows output, rowCohort row == Right index])
    unless (actualRows == expectedRows) (Source.invalid "Resident direct call records differ from the complete raw owner outputs")
    let equalCount = sum (map Resident.equalResults outputs)
        tokens = sum (map Resident.responseTokens outputs)
    unless (equalCount == equalResults finished && tokens == responseTokens finished && fromIntegral (length (Measurement.loadDurations measured)) == loads finished) (Source.invalid "Resident direct completion differs from its actual physical observations")
    pure Run.Run {Run.logDigest = Source.digest callSnapshot, Run.schedule = Run.Resident measured, Run.concurrent = length owners > 1, Run.criticalPath = Measurement.critical measured, Run.equalResults = equalCount, Run.completion = Just (Run.Completion (Source.digest snapshot) (wall finished) (process finished) (map provenance observed))}

rowCohort :: Value -> Either String Natural
rowCohort = parseEither (withObject "direct resident call" (.: "cohort"))

owner :: Source.Source -> FilePath -> (Natural, [Call.Call]) -> IO Observed
owner source root (index, expected) = do
    let prefix = root </> ("owner-" ++ padded index)
    (snapshot, fields) <- document source (prefix ++ ".status.json")
    status <- Source.checked (parseEither (Timing.status (index, length expected)) fields)
    checkStderr source (prefix ++ ".stderr.log") fields
    raw <- Source.snapshot source (prefix ++ ".stdout.jsonl")
    digest <- Source.checked (parseEither (\value -> value .: "stdout_sha256" >>= Json.identity) fields)
    unless (digest == Source.digest raw) (Source.invalid "Resident direct stdout identity mismatch")
    output <- Source.checked (Resident.admit (index, 0) expected (Source.encoded raw))
    pure (Observed status output (object ["owner" .= index, "status_sha256" .= Source.digest snapshot, "status_json" .= decodeUtf8 (Source.encoded snapshot)]))

complete :: (Replay.Reference, [(Natural, [Call.Call])]) -> Object -> Parser Complete
complete (reference, owners) fields = do
    Json.fields ["reference_log_sha256", "tasks_sha256", "policy", "mode", "sessions", "calls", "cohorts", "loads", "wall_seconds", "process_seconds", "response_tokens", "equal_results", "cohort_intervals", "close_intervals", "scope"] fields
    let evaluated = Replay.evaluation reference
        expected = Replay.calls reference
        indices = map fst (Call.cohorts expected)
    digest <- fields .: "reference_log_sha256"
    tasks <- fields .: "tasks_sha256"
    policy <- fields .: "policy"
    unless (digest == Evaluation.logDigest evaluated && tasks == Evaluation.inputDigest evaluated && policy == Evaluation.policy evaluated) (fail "Direct completion belongs to a different reference")
    mode <- fields .: "mode" :: Parser String
    count <- fields .: "calls" :: Parser Int
    cohorts <- fields .: "cohorts" :: Parser Int
    sessions <- fields .: "sessions" :: Parser Int
    unless (mode == "resident" && count == length expected && cohorts == length indices && sessions == length owners) (fail "Direct resident completion differs from the reference owner and cohort inventory")
    cohortIntervals <- fields .: "cohort_intervals" >>= traverse (Timing.interval "cohort")
    closeIntervals <- fields .: "close_intervals" >>= traverse (Timing.interval "owner")
    unless (map Timing.identity cohortIntervals == indices && map Timing.identity closeIntervals == reverse (map fst owners)) (fail "Direct resident interval identities differ from the reference")
    Complete <$> positive "wall_seconds" fields <*> positive "process_seconds" fields <*> fields .: "response_tokens" <*> fields .: "equal_results" <*> fields .: "loads" <*> pure (cohortIntervals, closeIntervals)
