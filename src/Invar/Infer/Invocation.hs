{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Invocation (Call, Permit, Error (..), prepare, arguments, input, batchInput, authorize, permission, loadFact, observe) where

import Control.Monad (foldM, unless)
import Data.Aeson (Object, eitherDecodeStrict, encode, (.:))
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8, encodeUtf8)
import Invar.Infer qualified as I
import Invar.Infer.Load qualified as Load
import Invar.Infer.Result qualified as R
import Invar.Infer.Wire qualified as Wire
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as L

data Call = Call I.Plan V.Binding ByteString V.Runtime Load.Plan

data Permit = Permit I.Plan V.Binding ByteString V.Runtime L.Fact ByteString

data Error = Preparation I.Error | Result R.Error | Protocol String | Lifecycle V.Error | Loading Load.Error | Registry L.Error
    deriving (Eq, Show)

prepare :: V.Binding -> I.Plan -> Either Error Call
prepare bound planned = do
    (program, runtime) <- either (Left . Preparation) Right (I.invocation bound planned)
    loading <- either (Left . Loading) Right (Load.prepare bound planned)
    pure (Call planned bound program runtime loading)

arguments :: Call -> [String]
arguments (Call planned _ _ _ _) = I.arguments planned

input :: Call -> String
input (Call _ bound program _ loading) = Text.unpack (decodeUtf8 (Lazy.toStrict encoded))
  where
    encoded = encode (Wire.envelopeValue (bound, program, Load.program loading))

batchInput :: Call -> ByteString
batchInput (Call planned bound program _ loading) = Lazy.toStrict (encode (Wire.batchValue (bound, program, Load.program loading) (I.requested planned)))

observe :: Permit -> ByteString -> Either Error (V.Completion, R.Result)
observe (Permit planned bound prefix consumed _ _) output = do
    unless (prefix `Bytes.isPrefixOf` output) (Left (Protocol "Completed stream differs from the authorized prefix"))
    result <- either (Left . Result) Right (R.observe planned output)
    final <- foldM (advance bound) consumed (Bytes.lines (Bytes.drop (Bytes.length prefix) output))
    completed <- lifecycle (V.completion final (V.boundAttempt bound))
    case completed of
        Just value -> Right (value, result)
        Nothing -> Left (Protocol "Worker output did not complete the bound invocation")

authorize :: L.Registry -> Call -> ByteString -> Either Error (L.Registry, Permit)
authorize registry (Call planned bound program prepared loading) output = do
    either (Left . Result) Right (R.ready planned output)
    updated <- either (Left . Loading) Right (Load.register loading registry output)
    live <- registryError (L.acquire updated (V.boundInstance bound))
    issued <- registryError (L.dispatch (L.Dispatch live bound) updated prepared)
    current <- foldM (advance bound) issued (Bytes.lines output)
    phase <- lifecycle (V.phase current (V.boundAttempt bound))
    unless (phase == V.Consumed) (Left (Protocol "Inference input has not been consumed"))
    fact <- registryError (L.historical updated (V.boundInstance bound))
    pure (updated, Permit planned bound output current fact (Lazy.toStrict (encode (Wire.invocationValue bound program))))

permission :: Permit -> ByteString
permission (Permit _ _ _ _ _ encoded) = encoded

loadFact :: Permit -> L.Fact
loadFact (Permit _ _ _ _ fact _) = fact

advance :: V.Binding -> V.Runtime -> ByteString -> Either Error V.Runtime
advance expected runtime encoded = do
    value <- either (Left . Protocol) Right (eitherDecodeStrict encoded)
    stage <- parse (.: "stage") value
    case stage :: String of
        "loaded_adapter" -> matching expected value >> pure runtime
        "consumed" -> do
            bound <- matching expected value
            program <- encodeUtf8 <$> parse (.: "program") value
            actual <- parse Wire.request value
            command <- either (Left . Preparation) Right (I.emission actual)
            lifecycle (V.consume (V.Consumption bound program command) runtime)
        "result" -> matching expected value >>= \bound -> lifecycle (V.finish bound encoded runtime)
        _ -> Right runtime

matching :: V.Binding -> Object -> Either Error V.Binding
matching expected value = do
    actual <- parse Wire.binding value
    unless (actual == expected) (Left (Lifecycle (V.BindingMismatch expected actual)))
    pure actual

parse :: (Object -> Parser value) -> Object -> Either Error value
parse parser = either (Left . Protocol) Right . parseEither parser

lifecycle :: Either V.Error value -> Either Error value
lifecycle = either (Left . Lifecycle) Right

registryError :: Either L.Error value -> Either Error value
registryError = either (Left . Registry) Right
