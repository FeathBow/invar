{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Result (Result, Error (..), observe, observeObjects, numerical, ready, record, restore, consumed, response, tokens, behavior, behaviorBits, referenceScores, promptLength, truncated) where

import Control.Monad (foldM, unless)
import Data.Aeson (FromJSON (parseJSON), Object, Value, eitherDecodeStrict, object, withObject, (.:), (.:?), (.=))
import Data.Aeson qualified as Json
import Data.Aeson.Types (Parser, parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Word (Word32)
import GHC.Float (castWord32ToFloat, float2Double)
import Invar.Infer qualified as I
import Invar.Infer.Output qualified as Output
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Fields
import Invar.Policy qualified as Policy
import Numeric.Natural (Natural)

data Report = Report
    { request :: I.Request
    , body :: Output.Body
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
        observed <- Output.parse value
        pure Report {request = input, body = observed}

observe :: I.Plan -> ByteString -> Either Error Result
observe planned encoded = traverse (either (Left . Malformed) Right . eitherDecodeStrict) (Bytes.lines encoded) >>= observeObjects planned

observeObjects :: I.Plan -> [Object] -> Either Error Result
observeObjects planned values = do
    final <- foldM (advance planned) Awaiting values
    case final of
        Finished result -> Right result
        _ -> Left (Unexpected "Worker output ended without a complete inference result")

numerical :: I.Plan -> ByteString -> Either Error Result
numerical planned encoded = do
    reported <- either (Left . Malformed) Right (eitherDecodeStrict encoded)
    result <- checked (I.requested planned) reported
    first Mismatch (Output.rawBehavior encoded (behaviorBits result))
    pure result

ready :: I.Plan -> ByteString -> Either Error ()
ready planned encoded = do
    values <- traverse (either (Left . Malformed) Right . eitherDecodeStrict) (Bytes.lines encoded)
    progress <- foldM (advance planned) Awaiting values
    case progress of
        Loaded -> Right ()
        _ -> Left (Unexpected "Inference consumption requires a matching adapter load report")

advance :: I.Plan -> Progress -> Object -> Either Error Progress
advance _ (Finished _) _ = Left (Unexpected "Output follows the completed inference result")
advance planned progress value = do
    stage <- parse (.: "stage") value
    case stage :: String of
        "loaded_adapter" -> loaded planned progress value
        "unloaded_adapter" -> case progress of
            Awaiting -> Right Awaiting
            _ -> Left (Unexpected "Unload follows the current adapter load")
        "result" -> finished (I.requested planned) progress value
        "consumed" -> case progress of
            Loaded -> Right Loaded
            _ -> Left (Unexpected "Consumption report arrived before the adapter load report")
        _ | stage `elem` ["loading", "profile", "load", "inference"] -> Right progress
        _ -> Left (Unexpected ("Unknown worker stage: " ++ stage))

parse :: (Object -> Parser value) -> Object -> Either Error value
parse parser = either (Left . Malformed) Right . parseEither parser

loaded :: I.Plan -> Progress -> Object -> Either Error Progress
loaded planned Awaiting value = do
    let expected = I.requested planned
    requested <- parse (.: "requested") value
    loadedIdentity <- parse (.: "consumed") value
    tokenizer <- parse (.: "tokenizer") value
    base <- parse (.: "base") value
    assembly <- parse (.: "assembly") value
    unless (requested == I.artifact expected && loadedIdentity == requested) (Left (Mismatch "Loaded adapter differs from the checked request"))
    unless (tokenizer == I.tokenizer expected) (Left (Mismatch "Loaded tokenizer differs from the checked request"))
    unless (base == I.base expected && assembly == I.assembly expected) (Left (Mismatch "Loaded model materialization differs from the checked request"))
    mapM_ checkSource (I.boundPolicy planned)
    pure Loaded
  where
    checkSource selected = do
        model <- parse (.: "model") value
        revision <- parse (.: "revision") value
        unless ((model, revision) == (Policy.model selected, Policy.revision selected)) (Left (Mismatch "Loaded model revision differs from the selected policy description"))
loaded _ _ _ = Left (Unexpected "Duplicate adapter load report")

finished :: I.Request -> Progress -> Object -> Either Error Progress
finished expected Loaded value = do
    reported <- parse (parseJSON . Json.Object) value
    Finished <$> checked expected reported
finished _ _ _ = Left (Unexpected "Inference result arrived before the adapter load report")

checked :: I.Request -> Report -> Either Error Result
checked expected reported = do
    unless (request reported == expected) (Left (Mismatch "Reported request differs from the checked emission"))
    first Mismatch (Output.validate (I.tokens (request reported)) (body reported))
    pure (Result reported)

record :: Result -> Value
record (Result reported) = object (Wire.requestPairs (request reported) ++ ["tokens" .= Output.tokens observed, "prompt_length" .= Output.prefix observed, "behavior_bits" .= Output.bits observed, "truncated" .= Output.truncated observed, "text" .= Output.decoded observed, "reference" .= fmap scored (Output.reference observed)])
  where
    observed = body reported
    scored chosen = object ["adapter" .= Output.adapter chosen, "bits" .= Output.scores chosen]

restore :: I.Request -> Value -> Either Error Result
restore expected encoded = do
    reported <- first Malformed (parseEither stored encoded)
    checked expected reported
  where
    stored = withObject "stored inference result" $ \value -> do
        Fields.fields ["adapter", "tokenizer", "base", "assembly", "request", "tokens", "prompt_length", "behavior_bits", "truncated", "text", "reference"] value
        bitWords <- value .: "behavior_bits"
        Report <$> Wire.request value <*> observed value bitWords
    observed value bitWords =
        Output.Body <$> value .: "tokens" <*> value .: "prompt_length" <*> pure (map (float2Double . castWord32ToFloat) bitWords) <*> pure bitWords <*> value .: "truncated" <*> value .: "text" <*> (value .:? "reference" >>= traverse scored)
    scored = withObject "stored reference scores" $ \value -> do
        Fields.fields ["adapter", "bits"] value
        Output.Scored <$> (value .: "adapter" >>= Fields.identity) <*> value .: "bits"

consumed :: Result -> I.Request
consumed (Result reported) = request reported

response :: Result -> String
response (Result reported) = Output.decoded (body reported)

tokens :: Result -> [Natural]
tokens (Result reported) = Output.tokens (body reported)

behavior :: Result -> [Double]
behavior (Result reported) = Output.probabilities (body reported)

behaviorBits :: Result -> [Word32]
behaviorBits (Result reported) = Output.bits (body reported)

referenceScores :: Result -> Maybe Output.Scored
referenceScores (Result reported) = Output.reference (body reported)

promptLength :: Result -> Natural
promptLength (Result reported) = Output.prefix (body reported)

truncated :: Result -> Bool
truncated (Result reported) = Output.truncated (body reported)
