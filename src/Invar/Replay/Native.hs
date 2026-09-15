{-# LANGUAGE OverloadedStrings #-}

module Invar.Replay.Native (plan, input, evaluate, quality) where

import Control.Monad (unless)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value, object, withObject, (.:), (.=))
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Foldable (traverse_)
import Data.Map.Strict qualified as Map
import Data.Text.Encoding (decodeUtf8)
import Invar.Artifact qualified as Artifact
import Invar.Cohort qualified as Cohort
import Invar.Infer.Result qualified as Result
import Invar.Json qualified as Json
import Invar.Learn qualified as Learn
import Invar.Learn.Observed qualified as Observed
import Invar.Quality.Summary qualified as Summary
import Invar.Reward qualified as Reward
import Invar.Workload qualified as Workload

plan :: Learn.Settings -> Workload.Document -> Either String Value
plan settings document = do
    first show (Learn.validate settings)
    traverse_ (Observed.task settings) (concatMap Workload.tasks (Workload.cycles document))
    pure (object ["format" .= ("invar-native-composition-v1" :: String), "workload" .= Workload.value document])

input :: Learn.Settings -> Workload.Cycle -> ByteString -> Either String Value
input settings workload encoded = do
    observed <- results settings workload =<< linesOf encoded
    (program, payload, rewards) <- Observed.input settings workload observed
    request <- Json.decode payload
    pure (object ["program" .= decodeUtf8 program, "request" .= request, "rewards" .= map (fromRational :: Rational -> Double) rewards, "scope" .= ("actual numerical observations; no live invocation or qualification authority" :: String)])

linesOf :: ByteString -> Either String [ByteString]
linesOf encoded = do
    unless ("\n" `Bytes.isSuffixOf` encoded) (Left "Incomplete native numerical result line")
    pure (Bytes.lines encoded)

results :: Learn.Settings -> Workload.Cycle -> [ByteString] -> Either String [Result.Result]
results settings workload encoded = do
    records <- traverse named encoded
    let indexed = Map.fromList records
        tasks = Workload.tasks workload
    unless (length records == Map.size indexed && Map.keysSet indexed == Map.keysSet (Map.fromList [(Workload.name task, ()) | task <- tasks])) (Left "Native result sample inventory differs from the declared cohort")
    traverse (readResult indexed) tasks
  where
    readResult indexed task = do
        declared <- Observed.task settings task
        bytes <- maybe (Left "Missing native sample") Right (Map.lookup (Workload.name task) indexed)
        first show (Result.numerical (Cohort.plan declared) bytes)

named :: ByteString -> Either String (String, ByteString)
named encoded = do
    value <- Json.decode encoded
    name <-
        parseEither
            ( withObject "native numerical result" $ \fields -> do
                Json.fields ["sample", "adapter", "tokenizer", "base", "assembly", "request", "tokens", "prompt_length", "behavior", "behavior_bits", "text", "truncated"] fields
                fields .: "request" >>= withObject "native numerical request" (Json.fields ["prompt", "tokens", "temperature", "seed"])
                fields .: "sample"
            )
            value
    pure (name, encoded)

evaluate :: Learn.Settings -> Workload.Document -> ByteString -> Either String Value
evaluate settings document encoded = do
    first show (Learn.validate settings)
    observed <- samples settings document =<< linesOf encoded
    measured <- Summary.measure observed
    pure (object ["policy" .= Learn.policy settings, "tasks_sha256" .= Workload.digest document, "log_sha256" .= Artifact.hex (SHA256.hash encoded), "overall" .= measured, "scope" .= ("core-scored complete numerical observations for one policy; no live evaluation, process-exit or qualification authority" :: String)])

quality :: (Learn.Settings, String) -> Workload.Document -> (ByteString, ByteString) -> Either String Value
quality (settings, trainedPolicy) document (initialLog, trainedLog) = do
    let trained = settings {Learn.policy = trainedPolicy}
    first show (Learn.validate settings)
    first show (Learn.validate trained)
    before <- samples settings document =<< linesOf initialLog
    after <- samples trained document =<< linesOf trainedLog
    measured <- Summary.compare before after
    pure (object (["comparison" .= ("paired native numerical observations" :: String), "tasks_sha256" .= Workload.digest document, "initial" .= source (Learn.policy settings) initialLog, "trained" .= source trainedPolicy trainedLog, "scope" .= ("core-scored complete numerical observations under one declared model profile; no live evaluation, process-exit, qualification or statistical-generalization authority" :: String)] ++ measured))
  where
    source policy encoded = object ["policy" .= policy, "log_sha256" .= Artifact.hex (SHA256.hash encoded)]

samples :: Learn.Settings -> Workload.Document -> [ByteString] -> Either String [Summary.Sample]
samples settings document = collect (zip [0 ..] (Workload.cycles document))
  where
    collect [] remaining = do
        unless (null remaining) (Left "Trailing native evaluation samples")
        pure []
    collect ((index, workload) : remaining) encoded = do
        let tasks = Workload.tasks workload
            (current, following) = splitAt (length tasks) encoded
        observed <- results settings workload current
        scored <- traverse (score index) (zip tasks observed)
        (scored ++) <$> collect remaining following
    score index (task, observed) = do
        reward <- first show (Reward.score (Workload.rule task) (Result.response observed) (Result.truncated observed))
        unless (Reward.value reward == 0 || Reward.value reward == 1) (Left "Expected the binary decimal-answer reward profile")
        pure (Summary.Sample index (Workload.name task) (Workload.group task) (Workload.seed task) (if Reward.value reward == 0 then 0 else 1) (fromIntegral (length (Result.behaviorBits observed))) (Result.truncated observed))
