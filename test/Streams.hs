{-# LANGUAGE OverloadedStrings #-}

module Streams (serial, batched) where

import Calls (change, field, request, setup, wire)
import Data.Aeson (Value (..), object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8)
import Hedgehog
import Invar.Infer qualified as Infer
import Invar.Workload qualified as Workload
import Workloads (encoded)

serial :: PropertyT IO (Workload.Document, [Value])
serial = do
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

batched :: PropertyT IO (Workload.Document, [Value], [Value])
batched = do
    (tasks, events) <- serial
    case events of
        [profile, load, firstLoaded, firstConsumed, timed, firstResult, _, secondLoaded, secondConsumed, _, secondResult, summary, completed] -> do
            let ready = frame "consumed" (map wire [[firstLoaded, firstConsumed], [secondLoaded, secondConsumed]])
                result = frame "result" (map (wire . pure) [firstResult, secondResult])
                grouped = [object ["stage" .= String "loading"], profile, load, ready, timed, result, summary, completed]
            pure (tasks, grouped, events)
        _ -> failure

frame :: Text -> [ByteString] -> Value
frame stage members = object ["stage" .= stage, "format" .= String "invar-inference-batch-v1", "calls" .= map decodeUtf8 members]

secondBinding :: Value
secondBinding = object ["call" .= Number 8, "attempt" .= Number 12, "instance" .= Number 14]

second :: Value -> Value
second (Object fields) = Object (Fields.insert "binding" secondBinding (Fields.mapWithKey update fields))
  where
    update "load" value = change "binding" secondBinding value
    update "request" value = change "seed" (Number 18) value
    update _ value = value
second value = value
