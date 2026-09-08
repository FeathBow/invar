{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Result (Result, Error (..), observe, ready, consumed, response, tokens, behavior, behaviorBits, promptLength, truncated) where

import Control.Monad (foldM, unless, when)
import Data.Aeson (FromJSON (parseJSON), Object, eitherDecodeStrict, withObject, (.:))
import Data.Aeson qualified as Json
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Word (Word32)
import GHC.Float (castWord32ToFloat, float2Double)
import Invar.Infer qualified as I
import Invar.Infer.Wire qualified as Wire
import Numeric.Natural (Natural)

data Report = Report
    { request :: I.Request
    , output :: [Natural]
    , prefix :: Natural
    , probabilities :: [Double]
    , probabilityBits :: [Word32]
    , limited :: Bool
    , decoded :: String
    }
    deriving (Eq, Show)

newtype Result = Result Report
    deriving (Eq, Show)

data Error = Malformed String | Unexpected String | Mismatch String
    deriving (Eq, Show)

data Progress = Awaiting | Loaded | Finished Result

instance FromJSON Report where
    parseJSON = withObject "inference result" $ \value -> do
        input <- Wire.request value
        tokenIds <- value .: "tokens"
        promptSize <- value .: "prompt_length"
        logprobs <- value .: "behavior"
        rawBits <- value .: "behavior_bits"
        wasLimited <- value .: "truncated"
        rendered <- value .: "text"
        pure Report {request = input, output = tokenIds, prefix = promptSize, probabilities = logprobs, probabilityBits = rawBits, limited = wasLimited, decoded = rendered}

observe :: I.Plan -> ByteString -> Either Error Result
observe planned encoded = do
    final <- foldM (advance (I.requested planned)) Awaiting (Bytes.lines encoded)
    case final of
        Finished result -> Right result
        _ -> Left (Unexpected "Worker output ended without a complete inference result")

ready :: I.Plan -> ByteString -> Either Error ()
ready planned encoded = do
    progress <- foldM (advance (I.requested planned)) Awaiting (Bytes.lines encoded)
    case progress of
        Loaded -> Right ()
        _ -> Left (Unexpected "Inference consumption requires a matching adapter load report")

advance :: I.Request -> Progress -> ByteString -> Either Error Progress
advance _ (Finished _) _ = Left (Unexpected "Output follows the completed inference result")
advance expected progress encoded = do
    value <- either (Left . Malformed) Right (eitherDecodeStrict encoded)
    stage <- parse (.: "stage") value
    case stage :: String of
        "loaded_adapter" -> loaded expected progress value
        "unloaded_adapter" -> case progress of
            Awaiting -> Right Awaiting
            _ -> Left (Unexpected "Unload follows the current adapter load")
        "result" -> finished expected progress value
        "consumed" -> case progress of
            Loaded -> Right Loaded
            _ -> Left (Unexpected "Consumption report arrived before the adapter load report")
        _ | stage `elem` ["loading", "profile", "load", "inference"] -> Right progress
        _ -> Left (Unexpected ("Unknown worker stage: " ++ stage))

parse :: (Object -> Parser value) -> Object -> Either Error value
parse parser = either (Left . Malformed) Right . parseEither parser

loaded :: I.Request -> Progress -> Object -> Either Error Progress
loaded expected Awaiting value = do
    requested <- parse (.: "requested") value
    loadedIdentity <- parse (.: "consumed") value
    tokenizer <- parse (.: "tokenizer") value
    base <- parse (.: "base") value
    assembly <- parse (.: "assembly") value
    unless (requested == I.artifact expected && loadedIdentity == requested) (Left (Mismatch "Loaded adapter differs from the checked request"))
    unless (tokenizer == I.tokenizer expected) (Left (Mismatch "Loaded tokenizer differs from the checked request"))
    unless (base == I.base expected && assembly == I.assembly expected) (Left (Mismatch "Loaded model materialization differs from the checked request"))
    pure Loaded
loaded _ _ _ = Left (Unexpected "Duplicate adapter load report")

finished :: I.Request -> Progress -> Object -> Either Error Progress
finished expected Loaded value = do
    reported <- parse (parseJSON . Json.Object) value
    unless (request reported == expected) (Left (Mismatch "Reported request differs from the checked emission"))
    validate reported
    pure (Finished (Result reported))
finished _ _ _ = Left (Unexpected "Inference result arrived before the adapter load report")

validate :: Report -> Either Error ()
validate reported = do
    let total = fromIntegral (length (output reported))
        count = fromIntegral (length (probabilities reported))
    unless (prefix reported > 0 && prefix reported < total) (Left (Mismatch "Invalid prompt boundary in generated tokens"))
    unless (count == total - prefix reported && count <= I.tokens (request reported)) (Left (Mismatch "Token and behavior-probability lengths disagree"))
    when (limited reported && count /= I.tokens (request reported)) (Left (Mismatch "Truncated output did not reach the checked token limit"))
    unless (all validProbability (probabilities reported)) (Left (Mismatch "Behavior log probabilities must be finite and nonpositive"))
    unless (length (probabilityBits reported) == length (probabilities reported) && and (zipWith corresponds (probabilities reported) (probabilityBits reported))) (Left (Mismatch "Behavior values and FP32 bits disagree"))
  where
    validProbability value = not (isNaN value || isInfinite value) && value <= 0
    corresponds value bits = value == float2Double (castWord32ToFloat bits)

consumed :: Result -> I.Request
consumed (Result reported) = request reported

response :: Result -> String
response (Result reported) = decoded reported

tokens :: Result -> [Natural]
tokens (Result reported) = output reported

behavior :: Result -> [Double]
behavior (Result reported) = probabilities reported

behaviorBits :: Result -> [Word32]
behaviorBits (Result reported) = probabilityBits reported

promptLength :: Result -> Natural
promptLength (Result reported) = prefix reported

truncated :: Result -> Bool
truncated (Result reported) = limited reported
