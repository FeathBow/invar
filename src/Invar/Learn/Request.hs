{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Request (Request, parse, value, logical) where

import Control.Monad (unless, when, (>=>))
import Data.Aeson (Value (Object), parseJSON, toJSON, withObject, (.:))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Word (Word32)
import GHC.Float (castWord32ToFloat)
import Invar.Json qualified as Json
import Numeric.Natural (Natural)

data Request = Request Value Value

value :: Request -> Value
value (Request original _) = original

logical :: Request -> Value
logical (Request _ ordered) = ordered

parse :: Value -> Parser Request
parse = withObject "numerical update request" $ \fields -> do
    Json.fields ["specification", "policy", "learner", "reference", "tokenizer", "base", "assembly", "behavior_model", "samples", "order", "epsilon", "penalty", "delta", "optimizer"] fields
    specification <- fields .: "specification"
    unless (specification == ("grpo-token-mean/v1" :: Text)) (fail "Unsupported update specification")
    mapM_ ((.:) fields >=> Json.identity) ["policy", "learner", "reference", "tokenizer", "base", "assembly"]
    fields .: "behavior_model" >>= model
    delivered <- fields .: "samples" >>= traverse sample
    order <- fields .: "order" :: Parser [Text]
    let named = Map.fromList [(name, original) | (name, _, original) <- delivered]
        grouped = Map.fromListWith (+) [(group, 1 :: Int) | (_, group, _) <- delivered]
    unless (not (null delivered) && Map.size named == length delivered) (fail "Cohort samples must be nonempty and distinct")
    unless (length order == Map.size named && Set.fromList order == Map.keysSet named) (fail "Logical order must name every admitted sample exactly once")
    unless (all (>= minimumGroup) grouped) (fail "Each advantage group requires at least two samples")
    ordered <- traverse (maybe (fail "Missing logical sample") pure . (`Map.lookup` named)) order
    epsilon <- fields .: "epsilon" >>= Json.finite
    penalty <- fields .: "penalty" >>= Json.finite
    delta <- fields .: "delta" >>= Json.finite
    unless (epsilon > 0 && epsilon < 1 && penalty >= 0 && delta > 0) (fail "Invalid GRPO coefficient configuration")
    fields .: "optimizer" >>= optimizer
    pure (Request (Object fields) (Object (Fields.insert "samples" (toJSON ordered) fields)))

model :: Value -> Parser ()
model = withObject "behavior model representation" $ \fields -> do
    Json.fields ["base", "assembly"] fields
    mapM_ ((.:) fields >=> Json.identity) ["base", "assembly"]

minimumGroup :: Int
minimumGroup = 2

sample :: Value -> Parser (Text, Text, Value)
sample original = withObject "update sample" inspect original
  where
    inspect fields = do
        Json.fields ["sample", "group", "prompt", "seed", "limit", "temperature", "tokens", "prompt_length", "behavior_bits", "text", "truncated", "reward", "advantage_bits"] fields
        name <- fields .: "sample"
        group <- fields .: "group"
        when (Text.null name || Text.null group) (fail "Logical sample and group identities must be nonempty")
        _ <- fields .: "prompt" :: Parser Text
        _ <- fields .: "text" :: Parser Text
        _ <- fields .: "seed" :: Parser Integer
        limit <- fields .: "limit" :: Parser Natural
        temperature <- fields .: "temperature" >>= Json.finite
        unless (limit > 0 && temperature > 0) (fail "Sample limit and temperature must be positive")
        tokens <- fields .: "tokens" :: Parser [Natural]
        prefix <- fields .: "prompt_length" :: Parser Natural
        words32 <- (fields .: "behavior_bits" :: Parser [Value]) >>= traverse (word True)
        let count = fromIntegral (length words32)
        unless (prefix > 0 && prefix < fromIntegral (length tokens) && count == fromIntegral (length tokens) - prefix && count <= limit) (fail "Behavior observations must match the admitted response tokens")
        truncated <- fields .: "truncated"
        unless (not truncated || count == limit) (fail "Invalid observed truncation status")
        _ <- fields .: "reward" >>= Json.finite
        _ <- fields .: "advantage_bits" >>= word False
        pure (name, group, original)

word :: Bool -> Value -> Parser Word32
word probability encoded = do
    decoded <- parseJSON encoded
    let number = castWord32ToFloat decoded
    unless (not (isNaN number || isInfinite number) && (not probability || number <= 0)) (fail "Expected finite FP32 words and nonpositive log probabilities")
    pure decoded

optimizer :: Value -> Parser ()
optimizer = withObject "AdamW configuration" $ \fields -> do
    Json.fields ["learning_rate", "betas", "epsilon", "weight_decay"] fields
    rate <- fields .: "learning_rate" >>= Json.finite
    betas <- (fields .: "betas" :: Parser [Value]) >>= traverse Json.finite
    epsilon <- fields .: "epsilon" >>= Json.finite
    decay <- fields .: "weight_decay" >>= Json.finite
    let moments = 2
    unless (rate >= 0 && epsilon > 0 && decay >= 0 && length betas == moments && all (\beta -> beta >= 0 && beta < 1) betas) (fail "Invalid AdamW scalar configuration")
