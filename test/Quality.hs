{-# LANGUAGE OverloadedStrings #-}

module Quality (quality) where

import Control.Monad (forM_)
import Data.Aeson (Value (..), toJSON)
import Data.Text qualified as Text
import Hedgehog
import Invar.Evaluation qualified as Evaluation
import Invar.Quality qualified as Quality
import Invar.Workload qualified as Workload
import Reports qualified
import Updates (alter, change, field)
import Workloads (array, declared, encoded)

quality :: Group
quality = Group "Core-owned quality comparisons" [("paired sample metrics use complete weighted denominators", once weighted), ("sample delivery cannot change quality comparisons", once reordered), ("truncated samples stay in all paired denominators", once truncated), ("quality comparison binds the complete input and model descriptions", once bindings)]
  where
    once = withTests 1 . property

trainedPolicy :: String
trainedPolicy = replicate 64 'b'

fixture :: PropertyT IO (Workload.Document, [Value], [Value])
fixture = do
    expected <- evalEither (Workload.decode (encoded declared))
    let initial = Reports.records Reports.policy expected
        first = change "summary" (change "zero_variance_groups" (Number 1) . change "reward_sum" (Number 2) $ field "summary" (at 0 initial)) . modifySample 0 (change "reward" (Number 1))
        second = change "summary" (change "zero_variance_groups" (Number 0) . change "reward_sum" (Number 1) $ field "summary" (at 1 initial)) . modifySample 1 (change "reward" (Number 1))
        trained = alter 0 first (alter 1 second (Reports.records trainedPolicy expected))
    pure (expected, initial, trained)

admit :: Workload.Document -> String -> [Value] -> PropertyT IO Evaluation.Report
admit expected policy records = evalEither (Evaluation.admit expected (Evaluation.Run policy 0) (Reports.stream records))

compareReports :: Workload.Document -> [Value] -> [Value] -> PropertyT IO Value
compareReports expected before after = do
    initial <- admit expected Reports.policy before
    trained <- admit expected trainedPolicy after
    evalEither (Quality.compare initial trained)

weighted :: PropertyT IO ()
weighted = do
    (expected, initial, trained) <- fixture
    result <- compareReports expected initial trained
    let overall = field "overall" result
    field "reward_sum" (field "initial" overall) === Number 1
    field "reward_sum" (field "trained" overall) === Number 3
    field "reward_mean" (field "initial" overall) === toJSON (1 / 6 :: Double)
    field "reward_mean_change" overall === toJSON (2 / 6 :: Double)
    map (`field` overall) ["improved_samples", "worsened_samples", "unchanged_samples"] === map Number [2, 0, 4]
    let groups = map (field "comparison") (array (field "by_group" result))
    map (field "sample_count" . field "initial") groups === map Number [2, 4]
    map (field "reward_mean_change") groups === map Number [0.5, 0.25]
    map (field "seed") (array (field "by_seed" result)) === map Number [17, 29, 43, 71]
    map (`field` field "zero_variance_groups" result) ["initial", "trained"] === map Number [1, 1]
    field "tasks_sha256" result === toJSON (Workload.digest expected)

reordered :: PropertyT IO ()
reordered = do
    (expected, initial, trained) <- fixture
    before <- compareReports expected initial trained
    let reversed row = change "samples" (toJSON (reverse (array (field "samples" row)))) row
    after <- compareReports expected initial (alter 0 reversed (alter 1 reversed trained))
    forM_ ["overall", "by_group", "by_seed"] $ \key -> field key before === field key after
    assert (field "log_sha256" (field "trained" before) /= field "log_sha256" (field "trained" after))

truncated :: PropertyT IO ()
truncated = do
    (expected, initial, trained) <- fixture
    let report = at 1 trained
        altered = alter 1 (change "summary" (change "truncated_count" (Number 1) (field "summary" report)) . modifySample 0 (change "truncated" (Bool True))) trained
    result <- compareReports expected initial altered
    let summary = field "trained" (field "overall" result)
    field "sample_count" summary === Number 6
    field "truncation_rate" summary === toJSON (1 / 6 :: Double)
    field "mean_response_tokens" summary === Number 4

bindings :: PropertyT IO ()
bindings = do
    (expected, initial, trained) <- fixture
    before <- admit expected Reports.policy initial
    changed <- evalEither (Workload.decode (encoded declared <> "\n"))
    other <- admit changed trainedPolicy (alter 2 (change "tasks_sha256" (toJSON (Workload.digest changed))) trained)
    rejected (Quality.compare before other)
    forM_ [[("tokenizer", String (Text.replicate 64 "c"))], [("tokenizer", String (Text.replicate 64 "c")), ("base", String (Text.replicate 64 "d")), ("assembly", String (Text.replicate 64 "e"))]] $ \description -> do
        let bound row = foldr (uncurry change) row description
        after <- admit expected trainedPolicy (alter 2 bound trained)
        rejected (Quality.compare before after)
        matching <- admit expected Reports.policy (alter 2 bound initial)
        _ <- evalEither (Quality.compare matching after)
        pure ()

modifySample :: Int -> (Value -> Value) -> Value -> Value
modifySample index operation cohort = change "samples" (toJSON (alter index operation (array (field "samples" cohort)))) cohort

at :: Int -> [value] -> value
at index values = case drop index values of
    value : _ -> value
    [] -> error "Missing fixed quality fixture record"

rejected :: (Show value) => Either String value -> PropertyT IO ()
rejected (Left _) = success
rejected (Right unexpected) = annotateShow unexpected >> failure
