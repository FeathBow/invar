{-# LANGUAGE OverloadedStrings #-}

module Invar.Score.Execution (Expected (..), observe) where

import Control.Monad (unless, when)
import Data.Aeson (FromJSON, Object, withObject, (.:))
import Data.Aeson.Key (Key)
import Data.Aeson.Types (Parser)
import Invar.Json qualified as Json
import Invar.Score.Probe (Relation (..))

data Expected = Expected {truncated :: Bool, responseTokens :: Int, fullVocabulary :: Bool}

observe :: Expected -> Object -> Parser Relation
observe expected fields = do
    engine <- fields .: "engine"
    case engine :: String of
        "mlx_lm.generate.BatchGenerator" -> LogOfMass <$ mlx expected fields
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner" -> SeparateLogSoftmax <$ vllm expected fields
        _ -> fail "Unsupported cached score execution engine"

mlx :: Expected -> Object -> Parser ()
mlx expected fields = do
    Json.fields ["engine", "sampling", "cache_origin", "unused_native_lookahead_draws", "truncated"] fields
    equals fields "cache_origin" ("fresh native caches; no reference cache input" :: String)
    equals fields "unused_native_lookahead_draws" (1 :: Int)
    equals fields "truncated" (truncated expected)
    fields .: "sampling"
        >>= withObject
            "native sampling"
            ( \settings -> do
                Json.fields ["batch_size", "prefill_step"] settings
                batch <- settings .: "batch_size"
                step <- settings .: "prefill_step"
                unless (batch > (0 :: Integer) && step > (0 :: Integer)) (fail "Invalid native batch/prefill sizes")
            )

vllm :: Expected -> Object -> Parser ()
vllm expected fields = do
    let additional = ["distribution" | fullVocabulary expected]
    Json.fields (["engine", "cache_origin", "path_control", "native_sample_rows", "ignored_prefill_rows", "truncated"] ++ additional) fields
    equals fields "cache_origin" ("target-owned native request caches; no source cache input" :: String)
    equals fields "path_control" ("replace sampled ids after native probability calculation" :: String)
    equals fields "truncated" (truncated expected)
    rows <- fields .: "native_sample_rows"
    ignored <- fields .: "ignored_prefill_rows"
    unless (ignored >= (0 :: Integer) && rows - ignored == fromIntegral (responseTokens expected)) (fail "Native scoring rows do not cover the prescribed response")
    when (fullVocabulary expected) (fields .: "distribution" >>= withObject "native mass/log measurement" distribution)

distribution :: Object -> Parser ()
distribution fields = do
    Json.fields ["mass_capture", "reported_log", "relation"] fields
    equals fields "mass_capture" ("torch.Tensor.softmax.output/F32/v1" :: String)
    equals fields "reported_log" ("torch.Tensor.log_softmax.output/F32/v1" :: String)
    equals fields "relation" ("separately rounded log-softmax and softmax/v1" :: String)

equals :: (Eq a, FromJSON a) => Object -> Key -> a -> Parser ()
equals fields key expected = do
    actual <- fields .: key
    unless (actual == expected) (fail ("Score execution differs at " ++ show key))
