{-# LANGUAGE OverloadedStrings #-}

module Reports (reports, records, outcomes, policy, stream) where

import Control.Monad (forM_)
import Data.Aeson (Key, Value (..), object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text qualified as Text
import Evaluations qualified
import Hedgehog
import Invar.Evaluation qualified as Evaluation
import Invar.Rollout qualified as Rollout
import Invar.Workload qualified as Workload
import Updates (alter, change, field)
import Workloads (array, declared, encoded, replace)

reports :: Group
reports = Group "Admitted evaluation reports" [("complete reports retain cohort-local samples and raw log identity", once complete), ("sample arrival order cannot select logical pairing", once reordered), ("reported summaries are independently recomputed", once summaries), ("sample domains and invocation inventories are checked", once malformed), ("every sample is joined to exactly one admitted inference of its task", once joined), ("missing duplicate reordered and trailing records fail", once terminal), ("successful completion cannot replace an observed successful exit", once process), ("exact input bytes bind every task field", once inputs), ("the completion names the complete materialization the inferences ran under", once materialization), ("finite sessions follow the declared session count and worker mode", once sessions), ("duplicate JSON keys in any worker or report record fail", once ambiguous)]
  where
    once = withTests 1 . property

policy :: String
policy = replicate 64 'a'

outcomes :: [[Evaluations.Outcome]]
outcomes = [[miss, hit], [miss, miss, miss, miss]]
  where
    miss = Evaluations.Outcome "#### 0" False
    hit = Evaluations.Outcome "#### 2" False

records :: String -> Workload.Document -> [[Evaluations.Outcome]] -> PropertyT IO [Value]
records selected expected = Evaluations.records (selected, Evaluations.identities) expected 1

stream :: [Value] -> ByteString
stream = Bytes.unlines . map encoded

fixture :: PropertyT IO (Workload.Document, [Value])
fixture = do
    expected <- evalEither (Workload.decode (encoded declared))
    values <- records policy expected outcomes
    pure (expected, values)

run :: Evaluation.Run
run = Evaluation.Run policy 0 Rollout.Serial Nothing

accepted :: (Workload.Document, [Value]) -> PropertyT IO Evaluation.Report
accepted (expected, values) = evalEither (Evaluation.admit expected run (stream values))

complete :: PropertyT IO ()
complete = do
    supplied@(expected, values) <- fixture
    result <- accepted supplied
    let (tokenizer, base, assembly) = Evaluations.identities
    Evaluation.inputDigest result === Workload.digest expected
    Evaluation.policy result === policy
    Evaluation.model result === Evaluation.Materialized tokenizer base assembly
    map Evaluation.sampleReward (Evaluation.samples result) === [0, 1, 0, 0, 0, 0]
    map Evaluation.sampleCohort (Evaluation.samples result) === [0, 0, 1, 1, 1, 1]
    map Evaluation.sampleName (Evaluation.samples result) === ["sample/17", "sample/29", "sample/17", "sample/29", "sample/43", "sample/71"]
    changed <- accepted (expected, alter 0 (change "cpu_seconds" (Number 3)) values)
    Evaluation.samples result === Evaluation.samples changed
    assert (Evaluation.logDigest result /= Evaluation.logDigest changed)

reordered :: PropertyT IO ()
reordered = do
    supplied@(expected, values) <- fixture
    initial <- accepted supplied
    let reverseSamples record = change "samples" (toJSON (reverse (array (field "samples" record)))) record
    arrived <- accepted (expected, Evaluations.phase 0 reverseSamples (Evaluations.phase 1 reverseSamples values))
    Evaluation.samples arrived === Evaluation.samples initial
    assert (Evaluation.logDigest arrived /= Evaluation.logDigest initial)

summaries :: PropertyT IO ()
summaries = do
    (expected, values) <- fixture
    let original = field "summary" (phaseRecord 0 values)
    forM_ ["sample_count", "reward_sum", "response_tokens", "truncated_count", "group_count", "zero_variance_groups"] $ \key -> do
        value <- evalMaybe (case field key original of Number number -> Just number; _ -> Nothing)
        reject expected (Evaluations.phase 0 (change "summary" (change key (Number (value + 1)) original)) values)
    reject expected (Evaluations.phase 0 (change "summary" (change "group_count" (Bool True) original)) values)

malformed :: PropertyT IO ()
malformed = do
    (expected, values) <- fixture
    forM_ [("seed", Number 18), ("seed", Bool True), ("name", String "unknown"), ("group", String "wrong"), ("reward", Bool True), ("reward", Number 0.5), ("response_tokens", Number 0), ("response_tokens", Number 5), ("response_tokens", Bool False), ("truncated", Number 1), ("extra", Null)] $ \(key, value) ->
        reject expected (Evaluations.phase 0 (modifySample 0 (change key value)) values)
    reject expected (Evaluations.phase 0 (modifySample 1 (change "truncated" (Bool True))) values)
    forM_ ["call", "attempt", "instance"] $ \axis -> do
        let original = field "binding" (at 0 (array (field "samples" (phaseRecord 1 values))))
        reject expected (Evaluations.phase 1 (modifySample 0 (change "binding" (change axis (Number 0) original))) values)
    let identity = field "binding" (at 0 (array (field "samples" (phaseRecord 0 values))))
    reject expected (Evaluations.phase 0 (modifySample 0 (change "binding" (change "extra" Null identity))) values)
    reject expected (Evaluations.phase 0 (change "policy" (String (Text.replicate 64 "b"))) values)

joined :: PropertyT IO ()
joined = do
    (expected, values) <- fixture
    let rewarded = Evaluations.phase 0 (change "summary" (change "reward_sum" (Number 2) (change "zero_variance_groups" (Number 1) (field "summary" (phaseRecord 0 values)))) . modifySample 0 (change "reward" (Number 1))) values
        truncatedSample = Evaluations.phase 1 (change "summary" (change "truncated_count" (Number 1) (field "summary" (phaseRecord 1 values))) . modifySample 0 (change "truncated" (Bool True))) values
        swapped = Evaluations.phase 0 (\record -> change "samples" (toJSON (swapBindings (array (field "samples" record)))) record) values
    forM_ [rewarded, truncatedSample, swapped] (reject expected)
  where
    swapBindings (first : second : rest) = change "binding" (field "binding" second) first : change "binding" (field "binding" first) second : rest
    swapBindings rest = rest

terminal :: PropertyT IO ()
terminal = do
    (expected, values) <- fixture
    let completed = at (length values - 1) values
        withoutCompletion = take (length values - 1) values
    forM_ [withoutCompletion, drop 1 values, values ++ [completed], withoutCompletion ++ [phaseRecord 0 values, completed], values ++ [object ["stage" .= String "profile"]], object ["stage" .= String "unknown"] : values, object ["phase" .= String "evaluation", "stage" .= String "result"] : values, Null : values] (reject expected)
    rejected (Evaluation.admit expected run (Bytes.init (stream values)))

process :: PropertyT IO ()
process = do
    (expected, values) <- fixture
    forM_ [1, -9] $ \status -> rejected (Evaluation.admit expected run {Evaluation.exitCode = status} (stream values))
    rejected (Evaluation.admit expected run {Evaluation.expectedPolicy = "invalid"} (stream values))
    rejected (Evaluation.admit expected run {Evaluation.workerMode = Rollout.Batched} (stream values))

inputs :: PropertyT IO ()
inputs = do
    (expected, values) <- fixture
    changed <- evalEither (Workload.decode (encoded declared <> "\n"))
    reject changed values
    forM_ [("Compute one plus one.", "Compute one plus two."), ("#### 2", "#### 3"), ("\"temperature\":0.8", "\"temperature\":0.7")] $ \(before, after) -> do
        input <- evalEither (Workload.decode (replace before after (encoded declared)))
        reject input values
    reject expected (Evaluations.completion (change "tasks_sha256" (String (Text.replicate 64 "0"))) values)

materialization :: PropertyT IO ()
materialization = do
    (expected, values) <- fixture
    forM_ ["tokenizer", "base", "assembly"] $ \axis -> do
        reject expected (Evaluations.completion (change axis (String (Text.replicate 64 "d"))) values)
        reject expected (Evaluations.completion (omit axis) values)
    other <- Evaluations.records (policy, (replicate 64 'c', replicate 64 'd', replicate 64 'f')) expected 1 outcomes
    result <- accepted (expected, other)
    Evaluation.model result === Evaluation.Materialized (replicate 64 'c') (replicate 64 'd') (replicate 64 'f')

sessions :: PropertyT IO ()
sessions = do
    (expected, _) <- fixture
    split <- Evaluations.records (policy, Evaluations.identities) expected 2 outcomes
    result <- accepted (expected, split)
    map Evaluation.sampleReward (Evaluation.samples result) === [0, 1, 0, 0, 0, 0]
    forM_ [Number 0, Number 1, Number 3, Number (-1), Number 1.5, Bool True, Null] $ \count -> reject expected (Evaluations.completion (change "sessions" count) split)

ambiguous :: PropertyT IO ()
ambiguous = do
    (expected, values) <- fixture
    forM_ ["{\"stage\":\"load\",\"stage\":\"profile\"}\n" <> stream values, replace "\"name\":" "\"name\":\"ignored\",\"name\":" (stream values), "{\"stage\":\n" <> stream values] $ \input ->
        rejected (Evaluation.admit expected run input)

phaseRecord :: Int -> [Value] -> Value
phaseRecord selected values = case drop selected [value | value@(Object fields) <- values, Fields.lookup "phase" fields == Just (String "evaluation")] of
    value : _ -> value
    [] -> error "Missing evaluation phase record"

omit :: Key -> Value -> Value
omit key (Object fields) = Object (Fields.delete key fields)
omit _ value = value

modifySample :: Int -> (Value -> Value) -> Value -> Value
modifySample index operation cohort = change "samples" (toJSON (alter index operation (array (field "samples" cohort)))) cohort

reject :: Workload.Document -> [Value] -> PropertyT IO ()
reject expected values = rejected (Evaluation.admit expected run (stream values))

rejected :: (Show value) => Either String value -> PropertyT IO ()
rejected (Left _) = success
rejected (Right unexpected) = annotateShow unexpected >> failure

at :: Int -> [value] -> value
at index values = case drop index values of
    value : _ -> value
    [] -> error "Missing fixed report fixture record"
