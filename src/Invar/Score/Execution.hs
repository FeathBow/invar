{-# LANGUAGE OverloadedStrings #-}

module Invar.Score.Execution (Expected (..), observe) where

import Control.Monad (unless, when)
import Data.Aeson (FromJSON, Object, withObject, (.:))
import Data.Aeson.Key (Key)
import Data.Aeson.Types (Parser)
import Invar.Json qualified as Json
import Invar.Score.Probe (Relation (..))

data Expected = Expected {responseTokens :: Int, fullVocabulary :: Bool}

-- Returns the probability relation and how many trailing response tokens the executor fed through the input embedding
-- beyond those scoring needs.
observe :: Expected -> Object -> Parser (Relation, Int)
observe expected fields = do
    engine <- fields .: "engine"
    case engine :: String of
        "mlx_lm.generate.BatchGenerator" -> (,) LogOfMass <$> mlx fields
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner" -> (SeparateLogSoftmax, 0) <$ vllm expected fields
        "transformers.forced_path" -> (LogOfMass, 0) <$ transformers expected fields
        _ -> fail "Unsupported cached score execution engine"

-- The engine loop feeds each token it yields, so the last response token reaches one more step whose draw is discarded.
mlx :: Object -> Parser Int
mlx fields = do
    Json.fields ["engine", "sampling", "cache_origin", "unused_native_lookahead_draws"] fields
    equals fields "cache_origin" ("fresh native caches; no reference cache input" :: String)
    lookahead <- fields .: "unused_native_lookahead_draws"
    unless (lookahead == 1) (fail "Native scoring must draw exactly one unused lookahead")
    fields .: "sampling"
        >>= withObject
            "native sampling"
            ( \settings -> do
                Json.fields ["batch_size", "prefill_step"] settings
                batch <- settings .: "batch_size"
                step <- settings .: "prefill_step"
                unless (batch > (0 :: Integer) && step > (0 :: Integer)) (fail "Invalid native batch/prefill sizes")
            )
    pure lookahead

vllm :: Expected -> Object -> Parser ()
vllm expected fields = do
    let additional = ["distribution" | fullVocabulary expected]
    Json.fields (["engine", "cache_origin", "path_control", "native_sample_rows", "ignored_prefill_rows"] ++ additional) fields
    equals fields "cache_origin" ("target-owned native request caches; no source cache input" :: String)
    equals fields "path_control" ("replace sampled ids after native probability calculation" :: String)
    rows <- fields .: "native_sample_rows"
    ignored <- fields .: "ignored_prefill_rows"
    unless (ignored >= (0 :: Integer) && rows - ignored == fromIntegral (responseTokens expected)) (fail "Native scoring rows do not cover the prescribed response")
    when (fullVocabulary expected) (fields .: "distribution" >>= withObject "native mass/log measurement" distribution)

transformers :: Expected -> Object -> Parser ()
transformers expected fields = do
    when (fullVocabulary expected) (fail "Hugging Face scoring records no full-vocabulary probe")
    Json.fields ["engine", "cache_origin"] fields
    equals fields "cache_origin" ("fresh model cache per path; no source cache input" :: String)

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
