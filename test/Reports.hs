{-# LANGUAGE OverloadedStrings #-}

module Reports (reports, records, policy, stream) where

import Control.Monad (forM_)
import Data.Aeson (Value (..), object, toJSON, (.=))
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text qualified as Text
import Hedgehog
import Invar.Evaluation qualified as Evaluation
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)
import Updates (alter, change, field)
import Workloads (array, declared, encoded, replace)

reports :: Group
reports = Group "Admitted evaluation reports" [("complete reports retain cohort-local samples and raw log identity", once complete), ("sample arrival order cannot select logical pairing", once reordered), ("reported summaries are independently recomputed", once summaries), ("sample domains and invocation inventories are checked", once malformed), ("missing duplicate reordered and trailing records fail", once terminal), ("successful completion cannot replace an observed successful exit", once process), ("exact input bytes bind every task field", once inputs), ("known worker stages and materialization alternatives remain explicit", once materialization), ("session declarations must match every cohort's actual load count", once sessions), ("duplicate JSON keys in any worker or report record fail", once ambiguous)]
  where
    once = withTests 1 . property

policy :: String
policy = replicate 64 'a'

records :: String -> Workload.Document -> [Value]
records selected expected =
    [cohort (0, 0) [17, 29] ([0, 1], 1, 0), cohort (1, 2) [17, 29, 43, 71] ([0, 0, 0, 0], 0, 1), object ["phase" .= String "evaluation_complete", "policy" .= selected, "cohorts" .= (2 :: Natural), "tasks_sha256" .= Workload.digest expected]]
  where
    cohort (index, offset) seeds (rewards, total, constants) = object ["phase" .= String "evaluation", "cohort" .= (index :: Natural), "policy" .= selected, "samples" .= zipWith3 sample seeds rewards [offset ..], "summary" .= object ["sample_count" .= length seeds, "reward_sum" .= (total :: Natural), "response_tokens" .= (4 * length seeds), "truncated_count" .= (0 :: Natural), "group_count" .= (1 :: Natural), "zero_variance_groups" .= (constants :: Natural)]]
    sample seed reward identity = object ["name" .= ("sample/" ++ show seed), "group" .= String "question", "seed" .= (seed :: Integer), "reward" .= (reward :: Natural), "response_tokens" .= (4 :: Natural), "truncated" .= False, "binding" .= object [key .= (identity :: Natural) | key <- ["call", "attempt", "instance"]]]

stream :: [Value] -> ByteString
stream = Bytes.unlines . map encoded

fixture :: PropertyT IO (Workload.Document, [Value])
fixture = do
    expected <- evalEither (Workload.decode (encoded declared))
    pure (expected, records policy expected)

accepted :: (Workload.Document, [Value]) -> PropertyT IO Evaluation.Report
accepted (expected, values) = evalEither (Evaluation.admit expected (Evaluation.Run policy 0) (stream values))

complete :: PropertyT IO ()
complete = do
    supplied@(expected, values) <- fixture
    result <- accepted supplied
    Evaluation.inputDigest result === Workload.digest expected
    Evaluation.policy result === policy
    Evaluation.model result === Evaluation.Unbound
    map Evaluation.sampleReward (Evaluation.samples result) === [0, 1, 0, 0, 0, 0]
    map Evaluation.sampleCohort (Evaluation.samples result) === [0, 0, 1, 1, 1, 1]
    map Evaluation.sampleName (Evaluation.samples result) === ["sample/17", "sample/29", "sample/17", "sample/29", "sample/43", "sample/71"]
    changed <- accepted (expected, object ["stage" .= String "profile"] : values)
    Evaluation.samples result === Evaluation.samples changed
    assert (Evaluation.logDigest result /= Evaluation.logDigest changed)

reordered :: PropertyT IO ()
reordered = do
    supplied@(expected, values) <- fixture
    initial <- accepted supplied
    let reverseSamples record = change "samples" (toJSON (reverse (array (field "samples" record)))) record
    arrived <- accepted (expected, alter 0 reverseSamples (alter 1 reverseSamples values))
    Evaluation.samples arrived === Evaluation.samples initial
    assert (Evaluation.logDigest arrived /= Evaluation.logDigest initial)

summaries :: PropertyT IO ()
summaries = do
    (expected, values) <- fixture
    let original = field "summary" (at 0 values)
    forM_ ["sample_count", "reward_sum", "response_tokens", "truncated_count", "group_count", "zero_variance_groups"] $ \key -> do
        value <- evalMaybe (case field key original of Number number -> Just number; _ -> Nothing)
        reject expected (alter 0 (change "summary" (change key (Number (value + 1)) original)) values)
    reject expected (alter 0 (change "summary" (change "group_count" (Bool True) original)) values)

malformed :: PropertyT IO ()
malformed = do
    (expected, values) <- fixture
    forM_ [("seed", Number 18), ("seed", Bool True), ("name", String "unknown"), ("group", String "wrong"), ("reward", Bool True), ("reward", Number 0.5), ("response_tokens", Number 0), ("response_tokens", Number 5), ("response_tokens", Bool False), ("truncated", Number 1), ("extra", Null)] $ \(key, value) ->
        reject expected (alter 0 (modifySample 0 (change key value)) values)
    reject expected (alter 0 (modifySample 1 (change "truncated" (Bool True))) values)
    forM_ ["call", "attempt", "instance"] $ \axis -> do
        let original = field "binding" (at 0 (array (field "samples" (at 1 values))))
        reject expected (alter 1 (modifySample 0 (change "binding" (change axis (Number 0) original))) values)
    let identity = field "binding" (at 0 (array (field "samples" (at 0 values))))
    reject expected (alter 0 (modifySample 0 (change "binding" (change "extra" Null identity))) values)
    reject expected (alter 0 (change "policy" (String (Text.replicate 64 "b"))) values)

terminal :: PropertyT IO ()
terminal = do
    (expected, values) <- fixture
    forM_ [take 2 values, drop 1 values, values ++ [at 2 values], [at 1 values, at 0 values, at 2 values], values ++ [object ["stage" .= String "profile"]], object ["stage" .= String "unknown"] : values, object ["phase" .= String "evaluation", "stage" .= String "result"] : values, Null : values] (reject expected)
    rejected (Evaluation.admit expected (Evaluation.Run policy 0) (Bytes.init (stream values)))

process :: PropertyT IO ()
process = do
    (expected, values) <- fixture
    forM_ [1, -9] $ \status -> rejected (Evaluation.admit expected (Evaluation.Run policy status) (stream values))
    rejected (Evaluation.admit expected (Evaluation.Run "invalid" 0) (stream values))

inputs :: PropertyT IO ()
inputs = do
    (expected, values) <- fixture
    changed <- evalEither (Workload.decode (encoded declared <> "\n"))
    reject changed values
    forM_ [("Compute one plus one.", "Compute one plus two."), ("#### 2", "#### 3"), ("\"temperature\":0.8", "\"temperature\":0.7")] $ \(before, after) -> do
        input <- evalEither (Workload.decode (replace before after (encoded declared)))
        reject input values
    reject expected (alter 2 (change "tasks_sha256" (String (Text.replicate 64 "0"))) values)

materialization :: PropertyT IO ()
materialization = do
    (expected, values) <- fixture
    forM_ [[], [("tokenizer", String (Text.replicate 64 "c"))], [("tokenizer", String (Text.replicate 64 "c")), ("base", String (Text.replicate 64 "d")), ("assembly", String (Text.replicate 64 "e"))]] $ \bindings -> do
        let extend value = foldr (uncurry change) value bindings
            worker = map (extend . object . pure . ("stage" .=)) (["loaded_adapter", "consumed", "result"] :: [String])
            observed = worker ++ alter 2 extend values
        result <- accepted (expected, observed)
        case bindings of
            [] -> Evaluation.model result === Evaluation.Unbound
            [_] -> Evaluation.model result === Evaluation.Tokenizer (replicate 64 'c')
            _ -> Evaluation.model result === Evaluation.Materialized (replicate 64 'c') (replicate 64 'd') (replicate 64 'e')
        reject expected (alter 0 (change "tokenizer" (String (Text.replicate 64 "f"))) observed)
    reject expected (alter 2 (change "base" (String (Text.replicate 64 "d"))) values)

sessions :: PropertyT IO ()
sessions = do
    (expected, values) <- fixture
    let load = object ["stage" .= String "load"]
        observed count = [load, load, at 0 values, load, load, at 1 values, change "sessions" count (at 2 values)]
    _ <- accepted (expected, observed (Number 2))
    forM_ [Number 0, Number 1, Number 3, Number (-1), Number 1.5, Bool True, Null] $ \count -> reject expected (observed count)
    reject expected [load, load, at 0 values, load, at 1 values, change "sessions" (Number 2) (at 2 values)]
    reject expected [at 0 values, at 1 values, load, at 2 values]

ambiguous :: PropertyT IO ()
ambiguous = do
    (expected, values) <- fixture
    forM_ ["{\"stage\":\"load\",\"stage\":\"profile\"}\n" <> stream values, replace "\"name\":" "\"name\":\"ignored\",\"name\":" (stream values), "{\"stage\":\n" <> stream values] $ \input ->
        rejected (Evaluation.admit expected (Evaluation.Run policy 0) input)

modifySample :: Int -> (Value -> Value) -> Value -> Value
modifySample index operation cohort = change "samples" (toJSON (alter index operation (array (field "samples" cohort)))) cohort

reject :: Workload.Document -> [Value] -> PropertyT IO ()
reject expected values = rejected (Evaluation.admit expected (Evaluation.Run policy 0) (stream values))

rejected :: (Show value) => Either String value -> PropertyT IO ()
rejected (Left _) = success
rejected (Right unexpected) = annotateShow unexpected >> failure

at :: Int -> [value] -> value
at index values = case drop index values of
    value : _ -> value
    [] -> error "Missing fixed report fixture record"
