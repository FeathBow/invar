{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Result (Result, Error (..), observe, ready, consumed, response, tokens, behavior, behaviorBits, referenceScores, promptLength, truncated) where

import Control.Monad (foldM, unless)
import Data.Aeson (FromJSON (parseJSON), Object, eitherDecodeStrict, withObject, (.:))
import Data.Aeson qualified as Json
import Data.Aeson.Types (Parser, parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Word (Word32)
import Invar.Float32 qualified as Float32
import Invar.Infer qualified as I
import Invar.Infer.Output qualified as Output
import Invar.Infer.Wire qualified as Wire
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

consumed :: Result -> I.Request
consumed (Result reported) = request reported

response :: Result -> String
response (Result reported) = Output.decoded (body reported)

tokens :: Result -> [Natural]
tokens (Result reported) = Output.tokens (body reported)

behavior :: Result -> [Double]
behavior (Result reported) = map Float32.double (Output.bits (body reported))

behaviorBits :: Result -> [Word32]
behaviorBits (Result reported) = Output.bits (body reported)

referenceScores :: Result -> Maybe Output.Scored
referenceScores (Result reported) = Output.reference (body reported)

promptLength :: Result -> Natural
promptLength (Result reported) = Output.prefix (body reported)

truncated :: Result -> Bool
truncated (Result reported) = Output.truncated (body reported)
