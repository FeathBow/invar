{-# LANGUAGE OverloadedStrings #-}

module Comparisons (comparisons) where

import BatchedObservations qualified
import Calls (change, field, request, wire)
import Control.Monad (forM_)
import Data.Aeson (Value (..), encode, object, toJSON, (.=))
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Hedgehog
import Invar.Evaluation qualified as Evaluation
import Invar.Infer qualified as Infer
import Invar.Measurement qualified as Measurement
import Invar.Measurement.Inference qualified as Inference
import Invar.Measurement.Manifest qualified as Manifest
import Invar.Measurement.Run qualified as Run
import Invar.Measurement.Statistics qualified as Statistics
import Invar.Spec.Load qualified as Load
import Invar.Workload qualified as Workload
import Measurements qualified
import System.IO.Error (ioeGetErrorString, tryIOError)
import Updates (alter)
import Workloads (array, everywhere)

comparisons :: Group
comparisons =
    Group
        "Repeated measurement comparisons"
        [ ("manifests reject failed incomplete or repeated runs", once manifests)
        , ("one retained log cannot be declared as two runs", once paths)
        , ("routes are compared by mean elapsed time and per run rates", once rates)
        , ("changed numerical results are retained and reported", once retained)
        , ("sessions need their own profile and load and one profile per comparison", once profiles)
        , ("concurrent sessions use the longest session per cohort", once sessions)
        , ("batched executions cannot be compared with serial executions", once batches)
        , ("materialized weights are part of the measured profile", once materialized)
        ]
  where
    once = withTests 1 . property

run :: Evaluation.Run
run = Evaluation.Run (Infer.artifact request) 0

entry :: Text -> Text -> Double -> Value
entry name route elapsed = object ["name" .= name, "route" .= route, "path" .= (name <> ".jsonl"), "exit_code" .= Number 0, "elapsed_seconds" .= elapsed]

first :: Value
first = entry "invar-0" "invar" 30

others :: [Value]
others = [entry "invar-1" "invar" 40, entry "direct-0" "direct" 20, entry "direct-1" "direct" 24]

encoded :: [Value] -> ByteString
encoded runs = Lazy.toStrict (encode (object ["reference_log" .= String "reference.jsonl", "reference_exit_code" .= Number 0, "runs" .= runs]))

declared :: PropertyT IO [Manifest.Run]
declared = evalEither (Manifest.runs <$> Manifest.admit (encoded (first : others)))

observe :: Workload.Document -> [Value] -> PropertyT IO Run.Run
observe tasks events = Inference.observedRun <$> evalEither (Inference.admit tasks run (wire events))

summaries :: Run.Run -> [Run.Run] -> PropertyT IO [Statistics.Summary]
summaries reference observed = do
    runs <- declared
    evalEither (traverse (\(selected, actual) -> Run.paired reference actual >>= Statistics.summarize selected) (zip runs observed))

rejected :: Either String value -> PropertyT IO ()
rejected = either (const success) (const failure)

manifests :: PropertyT IO ()
manifests = do
    runs <- declared
    map Manifest.name runs === ["invar-0", "invar-1", "direct-0", "direct-1"]
    forM_ [[], first : first : others, change "exit_code" (Number 1) first : others, change "elapsed_seconds" (Number 0) first : others, change "route" (String "native") first : others] $ \supplied ->
        rejected (Manifest.admit (encoded supplied))

paths :: PropertyT IO ()
paths = do
    (tasks, events) <- Measurements.fixture
    let files = Map.fromList [("reference.jsonl", wire events), ("manifest.json", encoded (first : change "name" (String "aliased") first : others))]
        source = Measurement.Source (\path -> maybe (ioError (userError path)) pure (Map.lookup path files)) pure
    outcome <- evalIO (tryIOError (Measurement.admit source tasks (Infer.artifact request, "manifest.json")))
    either (\problem -> ioeGetErrorString problem === "Repeated measurement path") (const failure) outcome

rates :: PropertyT IO ()
rates = do
    (tasks, events) <- Measurements.fixture
    reference <- observe tasks events
    runs <- declared
    summarized <- summaries reference (replicate 4 reference)
    result <- evalEither (Statistics.comparison summarized)
    field "invar_minus_direct_elapsed_seconds" result === Number 13
    field "all_results_equal_to_reference" result === Bool True
    forM_ (zip runs summarized) $ \(selected, summary) -> do
        let described = Statistics.describe summary
            elapsed = Manifest.elapsed selected
        field "response_tokens" described === Number 4
        field "response_tokens_per_elapsed_second" described === toJSON (4 / elapsed)
        field "worker_critical_path_seconds" described === Number 5
        field "seconds_outside_worker_critical_path" described === toJSON (elapsed - 5)
    forM_ [drop 1 summarized, take 3 summarized] (rejected . Statistics.comparison)

retained :: PropertyT IO ()
retained = do
    (tasks, events) <- Measurements.fixture
    reference <- observe tasks events
    changed <- observe tasks (alter 5 (change "tokens" (toJSON [1, 2, 4 :: Int])) events)
    summarized <- summaries reference (changed : replicate 3 reference)
    result <- evalEither (Statistics.comparison summarized)
    field "all_results_equal_to_reference" result === Bool False
    map (field "equal_results" . Statistics.describe) summarized === map Number [1, 2, 2, 2]

profiles :: PropertyT IO ()
profiles = do
    (tasks, events) <- Measurements.fixture
    reference <- observe tasks events
    rejected (Inference.admit tasks run (wire (take 1 events ++ drop 2 events)))
    changed <- observe tasks (alter 0 (change "precision" (String "different")) events)
    rejected (Run.paired reference changed)

partitioned :: [Value] -> PropertyT IO [Value]
partitioned events = case events of
    [profile, load, firstLoaded, firstConsumed, timed, firstResult, _, secondLoaded, secondConsumed, _, secondResult, report, finished] ->
        pure [profile, load, firstLoaded, firstConsumed, timed, firstResult, profile, load, secondLoaded, secondConsumed, timed, secondResult, report, change "sessions" (Number 2) finished]
    _ -> failure

sessions :: PropertyT IO ()
sessions = do
    (tasks, events) <- Measurements.fixture
    split <- partitioned events
    serial <- observe tasks events
    concurrent <- evalEither (Inference.admit tasks run (wire split))
    let reported = Inference.describe concurrent
        parallel = Inference.observedRun concurrent
    field "concurrent" reported === Bool True
    field "sessions_per_cohort" reported === toJSON [2 :: Int]
    field "critical_path_seconds" reported === Number 3
    summarized <- summaries parallel [parallel, parallel, serial, serial]
    either (=== "Matched measurements require the same number of model loads per cohort") (const failure) (Statistics.comparison summarized)

batches :: PropertyT IO ()
batches = do
    (tasks, grouped, serial) <- BatchedObservations.fixture
    batched <- observe tasks grouped
    separate <- observe tasks serial
    summarized <- summaries batched [batched, batched, separate, separate]
    either (=== "Matched measurements require the same request groups per numerical execution") (const failure) (Statistics.comparison summarized)

materialized :: PropertyT IO ()
materialized = do
    (tasks, events) <- Measurements.fixture
    let profile bytes = do
            report <- evalEither (Inference.admit tasks run bytes)
            case array (field "measurements" (Inference.describe report)) of
                measured : _ -> pure (field "profile_sha256" measured)
                [] -> failure
        zero = replicate 64 '0'
        rewritten changed = foldr (uncurry everywhere) (wire events) (zip (images request ++ digests request) (images changed ++ digests changed))
        images selected = let image = Infer.image selected in [Load.artifact image, Load.profile image]
        digests selected = map Bytes.pack [Infer.tokenizer selected, Infer.base selected, Infer.assembly selected]
    original <- profile (wire events)
    forM_ [request {Infer.tokenizer = zero}, request {Infer.base = zero}, request {Infer.assembly = zero}] $ \changed -> do
        measured <- profile (rewritten changed)
        measured /== original
