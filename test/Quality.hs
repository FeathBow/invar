{-# LANGUAGE OverloadedStrings #-}

module Quality (quality) where

import Control.Monad (forM_)
import Data.Aeson (Value (..), toJSON)
import Evaluations qualified
import Hedgehog
import Invar.Evaluation qualified as Evaluation
import Invar.Quality qualified as Quality
import Invar.Rollout qualified as Rollout
import Invar.Workload qualified as Workload
import Reports qualified
import Updates (change, field)
import Workloads (array, declared, encoded)

quality :: Group
quality = Group "Core-owned quality comparisons" [("paired sample metrics use complete weighted denominators", once weighted), ("sample delivery cannot change quality comparisons", once reordered), ("truncated samples stay in all paired denominators", once truncated), ("quality comparison binds the complete input and model descriptions", once bindings)]
  where
    once = withTests 1 . property

trainedPolicy :: String
trainedPolicy = replicate 64 'b'

hit, miss :: Evaluations.Outcome
hit = Evaluations.Outcome "#### 2" False
miss = Evaluations.Outcome "#### 0" False

trainedOutcomes :: [[Evaluations.Outcome]]
trainedOutcomes = [[hit, hit], [miss, hit, miss, miss]]

fixture :: PropertyT IO (Workload.Document, [Value], [Value])
fixture = do
    expected <- evalEither (Workload.decode (encoded declared))
    initial <- Reports.records Reports.policy expected Reports.outcomes
    trained <- Reports.records trainedPolicy expected trainedOutcomes
    pure (expected, initial, trained)

admit :: Workload.Document -> String -> [Value] -> PropertyT IO Evaluation.Report
admit expected policy records = evalEither (Evaluation.admit expected (Evaluation.Run policy 0 Rollout.Serial Nothing) (Reports.stream records))

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
    after <- compareReports expected initial (Evaluations.phase 0 reversed (Evaluations.phase 1 reversed trained))
    forM_ ["overall", "by_group", "by_seed"] $ \key -> field key before === field key after
    assert (field "log_sha256" (field "trained" before) /= field "log_sha256" (field "trained" after))

truncated :: PropertyT IO ()
truncated = do
    (expected, initial, _) <- fixture
    altered <- Reports.records trainedPolicy expected [[hit, hit], [Evaluations.Outcome "#### 0" True, hit, miss, miss]]
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
    otherTrained <- Reports.records trainedPolicy changed trainedOutcomes
    other <- admit changed trainedPolicy otherTrained
    rejected (Quality.compare before other)
    let materialization = (replicate 64 'c', replicate 64 'd', replicate 64 'f')
    after <- Evaluations.records (trainedPolicy, materialization) expected 1 trainedOutcomes >>= admit expected trainedPolicy
    rejected (Quality.compare before after)
    matching <- Evaluations.records (Reports.policy, materialization) expected 1 Reports.outcomes >>= admit expected Reports.policy
    _ <- evalEither (Quality.compare matching after)
    _ <- admit expected trainedPolicy trained
    pure ()

rejected :: (Show value) => Either String value -> PropertyT IO ()
rejected (Left _) = success
rejected (Right unexpected) = annotateShow unexpected >> failure
