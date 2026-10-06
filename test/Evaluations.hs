{-# LANGUAGE OverloadedStrings #-}

module Evaluations (Outcome (..), identities, records, phase, completion) where

import Calls (field)
import Control.Monad (foldM)
import Data.Aeson (Key, Value (..), eitherDecodeStrict, object, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString.Char8 qualified as Bytes
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text qualified as Text
import Data.Text.Encoding qualified as Text
import Data.Word (Word32)
import Hedgehog
import Invar.Infer qualified as Infer
import Invar.Infer.Invocation qualified as Call
import Invar.Reward qualified as Reward
import Invar.Spec.Invocation qualified as Invocation
import Invar.Spec.Load qualified as Load
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)
import Updates (alter)

data Outcome = Outcome {text :: String, truncated :: Bool}

identities :: (String, String, String)
identities = (replicate 64 'c', replicate 64 'e', replicate 64 'f')

records :: (String, (String, String, String)) -> Workload.Document -> Natural -> [[Outcome]] -> PropertyT IO [Value]
records (policy, (tokenizer, base, assembly)) expected sessions outcomes = do
    (_, cohorts) <- foldM cohort (0, []) (zip3 [0 :: Natural ..] (Workload.cycles expected) outcomes)
    pure (concat (reverse cohorts) ++ [object ["phase" .= String "evaluation_complete", "policy" .= policy, "tokenizer" .= tokenizer, "base" .= base, "assembly" .= assembly, "cohorts" .= length (Workload.cycles expected), "sessions" .= sessions, "tasks_sha256" .= Workload.digest expected]])
  where
    cohort (offset, accepted) (index, workload, results) = do
        let tasks = Workload.tasks workload
            count = fromIntegral sessions
            partitions = [[position | (order, position) <- zip [0 :: Natural ..] (Workload.order workload), order `mod` count == slot] | slot <- [0 .. count - 1]]
        calls <- traverse (\(position, task) -> prepared (offset + position) task) (zip [0 ..] tasks)
        executed <- traverse (session calls results) (filter (not . null) partitions)
        samples <- traverse (sample offset) (zip3 [0 ..] tasks results)
        pure (offset + fromIntegral (length tasks), (concat executed ++ [summarized index samples]) : accepted)
    prepared identity task = do
        planned <- evalEither (Infer.prepare (Infer.Request policy tokenizer base assembly (Workload.prompt task) (Workload.tokens task) (Workload.temperature task) (Workload.seed task)))
        called <- evalEither (Call.prepare (Invocation.ordinal identity) planned)
        envelope <- evalEither (eitherDecodeStrict (Text.encodeUtf8 (Text.pack (Call.input called))))
        pure (planned, identity, envelope)
    session calls results selected = do
        let members = [(calls !! fromIntegral position, results !! fromIntegral position) | position <- selected]
        pure (object ["stage" .= String "load", "cpu_seconds" .= Number 1] : concat (zipWith call (Nothing : map (Just . fst) members) members))
    call previous ((planned, identity, envelope), outcome) =
        [object ["stage" .= String "unloaded_adapter", "binding" .= field "binding" (field "load" earlier), "program" .= field "program" (field "load" earlier)] | Just (_, _, earlier) <- [previous]]
            ++ [loaded, consumed, object ["stage" .= String "inference", "cpu_seconds" .= Number 1], result]
      where
        requested = Infer.requested planned
        bound = object ["call" .= identity, "attempt" .= identity, "instance" .= identity]
        semantic = object ["prompt" .= Infer.prompt requested, "tokens" .= Infer.tokens requested, "temperature" .= Infer.temperature requested, "seed" .= Infer.seed requested]
        loading = field "load" envelope
        image = Infer.image requested
        materialization = ["tokenizer" .= tokenizer, "base" .= base, "assembly" .= assembly]
        loaded = object (materialization ++ ["stage" .= String "loaded_adapter", "binding" .= bound, "load" .= loading, "image" .= object ["artifact" .= Bytes.unpack (Load.artifact image), "profile" .= Bytes.unpack (Load.profile image)], "requested" .= policy, "consumed" .= policy, "model" .= String "test-model", "revision" .= String "test-revision"])
        consumed = object (materialization ++ ["stage" .= String "consumed", "binding" .= bound, "program" .= field "program" envelope, "load" .= loading, "adapter" .= policy, "request" .= semantic])
        result = object (materialization ++ ["stage" .= String "result", "binding" .= bound, "adapter" .= policy, "request" .= semantic, "tokens" .= [1, 2, 3, 4, 5 :: Int], "prompt_length" .= Number 1, "behavior" .= [-0.5, -0.25, -0.5, -0.25 :: Double], "behavior_bits" .= [0xbf000000, 0xbe800000, 0xbf000000, 0xbe800000 :: Word32], "text" .= text outcome, "truncated" .= truncated outcome])
    sample offset (position, task, outcome) = do
        scored <- evalEither (either (Left . show) Right (Reward.score (Workload.rule task) (text outcome) (truncated outcome)))
        let reward = if Reward.value scored == 0 then 0 else 1 :: Natural
            identity = offset + position
        pure (Workload.group task, reward, truncated outcome, object ["name" .= Workload.name task, "group" .= Workload.group task, "seed" .= Workload.seed task, "reward" .= reward, "response_tokens" .= (4 :: Natural), "truncated" .= truncated outcome, "binding" .= object [key .= identity | key <- ["call", "attempt", "instance" :: Key]]])
    summarized index samples =
        let groups = Map.fromListWith Set.union [(group, Set.singleton reward) | (group, reward, _, _) <- samples]
         in object
                [ "phase" .= String "evaluation"
                , "cohort" .= index
                , "policy" .= policy
                , "samples" .= [value | (_, _, _, value) <- samples]
                , "summary" .= object ["sample_count" .= length samples, "reward_sum" .= sum [reward | (_, reward, _, _) <- samples], "response_tokens" .= (4 * length samples), "truncated_count" .= length [() | (_, _, True, _) <- samples], "group_count" .= Map.size groups, "zero_variance_groups" .= length (filter ((== 1) . Set.size) (Map.elems groups))]
                ]

phase :: Int -> (Value -> Value) -> [Value] -> [Value]
phase selected modify values = alter position modify values
  where
    position = case drop selected [index | (index, Object fields) <- zip [0 ..] values, Fields.lookup "phase" fields == Just (String "evaluation")] of
        found : _ -> found
        [] -> error "Missing evaluation phase record"

completion :: (Value -> Value) -> [Value] -> [Value]
completion modify values = alter (length values - 1) modify values
