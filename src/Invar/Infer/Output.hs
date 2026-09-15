{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Output (Body (..), parse, validate, rawBehavior) where

import Control.Monad (unless, when)
import Data.Aeson (Object, (.:))
import Data.Aeson.Types (Parser)
import Data.ByteString (ByteString)
import Data.Word (Word32)
import GHC.Float (castDoubleToWord64, castWord32ToFloat, float2Double)
import Invar.Json qualified as Json
import Numeric.Natural (Natural)

data Body = Body
    { tokens :: [Natural]
    , prefix :: Natural
    , probabilities :: [Double]
    , bits :: [Word32]
    , truncated :: Bool
    , decoded :: String
    }
    deriving (Eq, Show)

parse :: Object -> Parser Body
parse fields = Body <$> fields .: "tokens" <*> fields .: "prompt_length" <*> fields .: "behavior" <*> fields .: "behavior_bits" <*> fields .: "truncated" <*> fields .: "text"

validate :: Natural -> Body -> Either String ()
validate limit body = do
    let total = fromIntegral (length (tokens body))
        count = fromIntegral (length (probabilities body))
    unless (prefix body > 0 && prefix body < total) (Left "Invalid prompt boundary in generated tokens")
    unless (count == total - prefix body && count <= limit) (Left "Token and behavior-probability lengths disagree")
    when (truncated body && count /= limit) (Left "Truncated output did not reach the checked token limit")
    unless (all validProbability (probabilities body)) (Left "Behavior log probabilities must be finite and nonpositive")
    unless (length (bits body) == length (probabilities body) && and (zipWith corresponds (probabilities body) (bits body))) (Left "Behavior values and FP32 bits disagree")
  where
    validProbability value = not (isNaN value || isInfinite value) && value <= 0
    corresponds value word = value == float2Double (castWord32ToFloat word)

rawBehavior :: ByteString -> [Word32] -> Either String ()
rawBehavior encoded bitWords = do
    reported <- map castDoubleToWord64 <$> Json.floatingArrayAt ["behavior"] encoded
    let actual = map (castDoubleToWord64 . float2Double . castWord32ToFloat) bitWords
    unless (reported == actual) (Left "Behavior value and FP32 word disagree, including zero sign")
