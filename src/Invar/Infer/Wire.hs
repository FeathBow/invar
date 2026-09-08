{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Wire (request, binding, bindingValue, invocationValue, envelopeValue, batchValue) where

import Data.Aeson (Object, Value, object, withObject, (.:), (.=))
import Data.Aeson.Types (Parser)
import Data.ByteString (ByteString)
import Data.Text.Encoding (decodeUtf8)
import Invar.Infer qualified as I
import Invar.Spec.Invocation qualified as V

request :: Object -> Parser I.Request
request value = do
    identity <- value .: "adapter"
    tokenizer <- value .: "tokenizer"
    base <- value .: "base"
    assembly <- value .: "assembly"
    value .: "request" >>= withObject "consumed request" (inputs (identity, tokenizer, base, assembly))
  where
    inputs (identity, tokenizer, base, assembly) fields = I.Request identity tokenizer base assembly <$> fields .: "prompt" <*> fields .: "tokens" <*> fields .: "temperature" <*> fields .: "seed"

binding :: Object -> Parser V.Binding
binding value = value .: "binding" >>= withObject "invocation binding" fields
  where
    fields identity = V.Binding . V.CallId <$> identity .: "call" <*> (V.AttemptId <$> identity .: "attempt") <*> (V.Instance <$> identity .: "instance")

bindingValue :: V.Binding -> Value
bindingValue (V.Binding (V.CallId call) (V.AttemptId attempt) (V.Instance instanceName)) = object ["call" .= call, "attempt" .= attempt, "instance" .= instanceName]

invocationValue :: V.Binding -> ByteString -> Value
invocationValue bound program = object ["binding" .= bindingValue bound, "program" .= decodeUtf8 program]

envelopeValue :: (V.Binding, ByteString, ByteString) -> Value
envelopeValue (bound, program, loadProgram) = object ["binding" .= bindingValue bound, "program" .= decodeUtf8 program, "load" .= invocationValue bound loadProgram]

batchValue :: (V.Binding, ByteString, ByteString) -> I.Request -> Value
batchValue (bound, program, loadProgram) requested =
    object
        [ "binding" .= bindingValue bound
        , "program" .= decodeUtf8 program
        , "load" .= invocationValue bound loadProgram
        , "adapter" .= I.artifact requested
        , "tokenizer" .= I.tokenizer requested
        , "base" .= I.base requested
        , "assembly" .= I.assembly requested
        , "request" .= object ["prompt" .= I.prompt requested, "tokens" .= I.tokens requested, "temperature" .= I.temperature requested, "seed" .= I.seed requested]
        ]
