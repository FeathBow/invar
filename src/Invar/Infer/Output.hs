{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Output (Body (..), Scored (..), parse, validate, rawBehavior) where

import Control.Monad (unless, when)
import Data.Aeson (Object, withObject, (.:), (.:?))
import Data.Aeson.Types (Parser)
import Data.ByteString (ByteString)
import Data.Word (Word32)
import GHC.Float (castDoubleToWord64)
import Invar.Float32 qualified as Float32
import Invar.Json qualified as Json
import Numeric.Natural (Natural)

data Body = Body
    { tokens :: [Natural]
    , prefix :: Natural
    , probabilities :: [Double]
    , bits :: [Word32]
    , truncated :: Bool
    , decoded :: String
    , reference :: Maybe Scored
    }
    deriving (Eq, Show)

data Scored = Scored {adapter :: String, scores :: [Word32]}
    deriving (Eq, Show)

parse :: Object -> Parser Body
parse fields = Body <$> fields .: "tokens" <*> fields .: "prompt_length" <*> fields .: "behavior" <*> fields .: "behavior_bits" <*> fields .: "truncated" <*> fields .: "text" <*> (fields .:? "reference" >>= traverse scored)
  where
    scored = withObject "reference scores" $ \values -> do
        Json.fields ["adapter", "bits"] values
        Scored <$> (values .: "adapter" >>= Json.identity) <*> values .: "bits"

validate :: Natural -> Body -> Either String ()
validate limit body = do
    let total = fromIntegral (length (tokens body))
        count = fromIntegral (length (probabilities body))
    unless (prefix body > 0 && prefix body < total) (Left "Invalid prompt boundary in generated tokens")
    unless (count == total - prefix body && count <= limit) (Left "Token and behavior-probability lengths disagree")
    when (truncated body && count /= limit) (Left "Truncated output did not reach the checked token limit")
    unless (all validProbability (probabilities body)) (Left "Behavior log probabilities must be finite and nonpositive")
    unless (length (bits body) == length (probabilities body) && and (zipWith corresponds (probabilities body) (bits body))) (Left "Behavior values and FP32 bits disagree")
    unless (all (\scored -> length (scores scored) == length (bits body) && all Float32.logProbability (scores scored)) (reference body)) (Left "Reference scores must be log probabilities of every response token")
  where
    validProbability value = not (isNaN value || isInfinite value) && value <= 0
    corresponds value word = value == Float32.double word

rawBehavior :: ByteString -> [Word32] -> Either String ()
rawBehavior encoded bitWords = do
    reported <- map castDoubleToWord64 <$> Json.floatingArrayAt ["behavior"] encoded
    let actual = map Float32.widened bitWords
    unless (reported == actual) (Left "Behavior value and FP32 word disagree, including zero sign")
