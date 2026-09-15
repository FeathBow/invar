{-# LANGUAGE OverloadedStrings #-}

module Measurements (measurements, fixture) where

import Calls (change, field, request, setup, wire)
import Control.Monad (forM_)
import Data.Aeson (Value (..), object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Hedgehog
import Invar.Evaluation qualified as Evaluation
import Invar.Infer qualified as Infer
import Invar.Measurement.Inference qualified as Measurement
import Invar.Replay.Inference qualified as Replay
import Invar.Workload qualified as Workload
import Updates (alter)
import Workloads (encoded)

measurements :: Group
measurements = Group "Reported inference measurements" [("measured observations retain CPU durations and cohort load counts", once complete), ("replay validity cannot supply missing measurement profiles", once profiles), ("loaded images summaries and duration kinds are checked", once correspondence)]
  where
    once = withTests 1 . property

fixture :: PropertyT IO (Workload.Document, [Value])
fixture = do
    (_, events) <- setup
    let task = object ["name" .= String "sample", "group" .= String "question", "prompt" .= Infer.prompt request, "seed" .= Infer.seed request, "tokens" .= Infer.tokens request, "temperature" .= Infer.temperature request, "answer" .= String "#### 12"]
        secondTask = change "name" (String "another") (change "seed" (Number 18) task)
        cohort = object ["tasks" .= [task, secondTask], "order" .= [0, 1 :: Int], "delivery" .= [0, 1 :: Int]]
    tasks <- evalEither (Workload.decode (encoded (toJSON [cohort])))
    case events of
        [loaded, consumed, result] -> do
            let summary = object ["sample_count" .= Number 2, "reward_sum" .= Number 0, "response_tokens" .= Number 4, "truncated_count" .= Number 2, "group_count" .= Number 1, "zero_variance_groups" .= Number 1]
                sample = object ["name" .= String "sample", "group" .= String "question", "seed" .= Infer.seed request, "reward" .= Number 0, "response_tokens" .= Number 2, "truncated" .= True, "binding" .= field "binding" consumed]
                secondSample = change "name" (String "another") (change "seed" (Number 18) (change "binding" secondBinding sample))
                report = object ["phase" .= String "evaluation", "cohort" .= Number 0, "policy" .= Infer.artifact request, "samples" .= [sample, secondSample], "summary" .= summary]
                finished = object ["phase" .= String "evaluation_complete", "cohorts" .= Number 1, "sessions" .= Number 1, "policy" .= Infer.artifact request, "tasks_sha256" .= Workload.digest tasks, "tokenizer" .= Infer.tokenizer request, "base" .= Infer.base request, "assembly" .= Infer.assembly request]
                timed = object ["stage" .= String "inference", "cpu_seconds" .= Number 2]
                materialized = change "model" (String "fixture") (change "revision" (String "fixture") loaded)
                unloaded = change "stage" (String "unloaded_adapter") (field "load" consumed)
            pure (tasks, [object ["stage" .= String "profile", "precision" .= String "protocol fixture"], object ["stage" .= String "load", "cpu_seconds" .= Number 1], materialized, consumed, timed, result, unloaded, second materialized, second consumed, timed, second result, report, finished])
        _ -> failure

secondBinding :: Value
secondBinding = object ["call" .= Number 8, "attempt" .= Number 12, "instance" .= Number 14]

second :: Value -> Value
second (Object fields) = Object (Fields.insert "binding" secondBinding (Fields.mapWithKey update fields))
  where
    update "load" value = change "binding" secondBinding value
    update "request" value = change "seed" (Number 18) value
    update _ value = value
second value = value

run :: Evaluation.Run
run = Evaluation.Run (Infer.artifact request) 0

complete :: PropertyT IO ()
complete = do
    (tasks, events) <- fixture
    observed <- evalEither (Measurement.admit tasks run (wire events))
    let result = Measurement.describe observed
    field "cohorts" result === Number 1
    field "sessions_per_cohort" result === toJSON [1 :: Int]
    field "critical_path_seconds" result === Number 5
    field "concurrent" result === Bool False

profiles :: PropertyT IO ()
profiles = do
    (tasks, events) <- fixture
    let unreported = wire (drop 1 events)
    reference <- evalEither (Replay.admit tasks (run, Replay.Session) unreported)
    length (Replay.calls reference) === 2
    reject tasks (drop 1 events)

correspondence :: PropertyT IO ()
correspondence = do
    (tasks, events) <- fixture
    forM_ ["artifact", "profile"] $ \key -> reject tasks (alter 2 (\value -> change "image" (change key (String "wrong") (field "image" value)) value) events)
    reject tasks (alter 11 (\value -> change "summary" (change "response_tokens" (Number 3) (field "summary" value)) value) events)
    let gpu = object ["stage" .= String "inference", "seconds" .= Number 2, "peak_allocated" .= Number 0, "peak_reserved" .= Number 0]
    reject tasks (alter 4 (const gpu) events)

reject :: Workload.Document -> [Value] -> PropertyT IO ()
reject tasks events = case Measurement.admit tasks run (wire events) of
    Left _ -> success
    Right result -> annotateShow (Measurement.describe result) >> failure
