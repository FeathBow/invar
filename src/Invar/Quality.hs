{-# LANGUAGE OverloadedStrings #-}

module Invar.Quality (compare) where

import Control.Monad (unless)
import Data.Aeson (Value, object, (.=))
import Invar.Evaluation qualified as Evaluation
import Invar.Quality.Summary qualified as Summary
import Prelude hiding (compare)

compare :: Evaluation.Report -> Evaluation.Report -> Either String Value
compare first second = do
    unless (Evaluation.inputDigest first == Evaluation.inputDigest second) (Left "Compared evaluations must use the same frozen input")
    unless (Evaluation.model first == Evaluation.model second) (Left "Compared evaluations must have the same reported model and tokenizer bindings")
    measured <- Summary.compare (map sample (Evaluation.samples first)) (map sample (Evaluation.samples second))
    pure (object (["comparison" .= ("paired evaluation summaries" :: String), "tasks_sha256" .= Evaluation.inputDigest first, "initial" .= source first, "trained" .= source second, "scope" .= ("complete reported evaluations and supplied process exit status; not report authenticity, numerical qualification, or statistical generalization" :: String)] ++ measured))

source :: Evaluation.Report -> Value
source report = object ["policy" .= Evaluation.policy report, "log_sha256" .= Evaluation.logDigest report, "exit_code" .= (0 :: Int)]

sample :: Evaluation.Sample -> Summary.Sample
sample value = Summary.Sample (Evaluation.sampleCohort value) (Evaluation.sampleName value) (Evaluation.sampleGroup value) (Evaluation.sampleSeed value) (Evaluation.sampleReward value) (Evaluation.sampleTokens value) (Evaluation.sampleTruncated value)
