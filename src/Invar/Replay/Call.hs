{-# LANGUAGE OverloadedStrings #-}

module Invar.Replay.Call (Call, admit, decode, value, cohort, cohorts, consumed, result, rawConsumed, bound, responseTokens, session) where

import Control.Monad (unless, when)
import Data.Aeson (Object, Value (..), encode, object, withObject, (.:), (.=))
import Data.Aeson.Encoding qualified as Encoding
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Lazy qualified as Lazy
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8, encodeUtf8)
import GHC.Float (castWord32ToFloat, float2Double)
import Invar.Infer.Model qualified as Model
import Invar.Infer.Output qualified as Output
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Spec.Invocation qualified as V
import Numeric.Natural (Natural)

data Request = Request String Natural Double Integer

data Call = Call
    { cohort :: Natural
    , consumed :: Object
    , result :: Object
    , rawConsumed :: ByteString
    , bound :: V.Binding
    , body :: Output.Body
    , request :: Request
    , loadProgram :: Maybe Text.Text
    }

admit :: Natural -> (ByteString, ByteString) -> Either String Call
admit index (first, lastOutput) = do
    input <- Json.decode first >>= parseEither (withObject "replay consumption" pure)
    finished <- Json.decode lastOutput >>= parseEither (withObject "replay result" pure)
    (binding, requested@(Request _ limit _ _), selected) <- parseEither consumption input
    observed <- parseEither (output input selected) finished
    Output.validate limit observed
    Output.rawBehavior lastOutput (Output.bits observed)
    loading <- parseEither (\fields -> if Fields.member "load" fields then Just <$> (fields .: "load" >>= (.: "program")) else pure Nothing) input
    pure (Call index input finished first binding observed requested loading)

consumption :: Object -> Parser (V.Binding, Request, Model.Model)
consumption fields = do
    selected <- Model.binding fields
    let loading = Fields.member "load" fields
    Json.fields (["stage", "binding", "program", "adapter", "request"] ++ Model.fields selected ++ ["load" | loading]) fields
    stage <- fields .: "stage" :: Parser String
    unless (stage == "consumed") (fail "Expected a consumed inference record")
    binding <- invocation fields
    _ <- fields .: "adapter" >>= Json.identity
    requested <- fields .: "request" >>= numerical
    when loading $ do
        actual <- fields .: "load" >>= withObject "replay load invocation" (\loadingFields -> Json.fields ["binding", "program"] loadingFields >> invocation loadingFields)
        unless (actual == binding) (fail "Reference load and inference bindings differ")
    pure (binding, requested, selected)

invocation :: Object -> Parser V.Binding
invocation fields = do
    fields .: "binding" >>= withObject "replay binding" (Json.fields ["call", "attempt", "instance"])
    program <- fields .: "program"
    when (Text.null program) (fail "Missing consumed program")
    Wire.binding fields

numerical :: Value -> Parser Request
numerical = withObject "replay numerical request" $ \fields -> do
    Json.fields ["prompt", "seed", "tokens", "temperature"] fields
    prompt <- fields .: "prompt" :: Parser String
    seed <- fields .: "seed"
    count <- fields .: "tokens"
    thermal <- fields .: "temperature" >>= Json.finite
    unless (count > 0 && thermal > 0 && '\0' `notElem` prompt) (fail "Invalid inference prompt, token budget or temperature")
    pure (Request prompt count thermal seed)

output :: Object -> Model.Model -> Object -> Parser Output.Body
output expected selected fields = do
    actual <- Model.binding fields
    unless (actual == selected) (fail "Direct model and tokenizer binding differs from the reference")
    Json.fields (["stage", "binding", "adapter", "request", "tokens", "prompt_length", "behavior", "behavior_bits", "text", "truncated"] ++ Model.fields selected) fields
    stage <- fields .: "stage" :: Parser String
    unless (stage == "result") (fail "Expected an inference result record")
    unless (all (\key -> Fields.lookup key fields == Fields.lookup key expected) ["binding", "adapter", "request"]) (fail "Direct result binding differs from the reference")
    Output.parse fields

decode :: Value -> Either String Call
decode encoded = do
    (index, first, lastOutput) <- parseEither (withObject "replay call" parse) encoded
    admit index (encodeUtf8 first, encodeUtf8 lastOutput)
  where
    parse fields = do
        Json.fields ["cohort", "consumed_json", "result_json"] fields
        (,,) <$> fields .: "cohort" <*> fields .: "consumed_json" <*> fields .: "result_json"

value :: Call -> Value
value call = object ["cohort" .= cohort call, "consumed_json" .= decodeUtf8 first, "result_json" .= decodeUtf8 lastOutput]
  where
    requested = case request call of
        Request prompt limit thermal seed -> object ["prompt" .= prompt, "tokens" .= limit, "temperature" .= thermal, "seed" .= seed]
    binding = Wire.bindingValue (bound call)
    common = Fields.insert "binding" binding . Fields.insert "request" requested
    input = common (consumed call)
    withLoad = maybe input (\program -> Fields.insert "load" (object ["binding" .= binding, "program" .= program]) input) (loadProgram call)
    first = Lazy.toStrict (encode (Object withLoad))
    outputFields = Fields.delete "behavior" (Fields.union (Fields.fromList ["tokens" .= Output.tokens (body call), "prompt_length" .= Output.prefix (body call), "behavior_bits" .= Output.bits (body call)]) (common (result call)))
    probabilityValues = map (float2Double . castWord32ToFloat) (Output.bits (body call))
    lastOutput = Lazy.toStrict (Encoding.encodingToLazyByteString (Encoding.pairs ("behavior" .= probabilityValues <> foldMap (uncurry (.=)) (Fields.toList outputFields))))

session :: Call -> Either String ()
session call = do
    selected <- parseEither Model.binding (consumed call)
    case selected of
        Model.Materialized {} | Fields.member "load" (consumed call) -> pure ()
        _ -> Left "A session replay requires a batch reference with load and materialization bindings"

responseTokens :: Call -> Natural
responseTokens = fromIntegral . length . Output.bits . body

cohorts :: [Call] -> [(Natural, [Call])]
cohorts = foldr collect []
  where
    collect call ((index, members) : rest) | cohort call == index = (index, call : members) : rest
    collect call rest = (cohort call, [call]) : rest
