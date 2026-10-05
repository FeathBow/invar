{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Records (Identities (..), loaded, consumed, result, unloaded, profile) where

import Control.Monad (unless, when)
import Data.Aeson (Object, Value (..), object, withObject, (.:), (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text qualified as Text
import Invar.Infer qualified as Infer
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as Load

data Identities = Identities {requested :: String, adapter :: String, tokenizer :: String, base :: String, assembly :: String, model :: String, revision :: String}
    deriving (Eq, Show)

loaded :: (Infer.Plan, V.Binding, Object) -> Object -> Either String Identities
loaded (planned, bound, expected) = parseEither $ \fields -> do
    let required = ["stage", "binding", "load", "image", "requested", "consumed", "tokenizer", "base", "assembly", "model", "revision"]
        selected = Infer.requested planned
        image = Infer.image selected
        imageValue = object ["artifact" .= Bytes.unpack (Load.artifact image), "profile" .= Bytes.unpack (Load.profile image)]
    Json.fields (required ++ ["scope" | Fields.member "scope" fields]) fields
    binding bound fields
    loading <- fields .: "load"
    unless (Just loading == Fields.lookup "load" expected) (fail "Inference load invocation differs from consumption")
    actual <- fields .: "image"
    unless (actual == imageValue) (fail "Inference load image differs from the declared materialization")
    found <- Identities <$> identity fields "requested" <*> identity fields "consumed" <*> identity fields "tokenizer" <*> identity fields "base" <*> identity fields "assembly" <*> fields .: "model" <*> fields .: "revision"
    unless ((requested found, adapter found, tokenizer found, base found, assembly found) == (Infer.artifact selected, Infer.artifact selected, Infer.tokenizer selected, Infer.base selected, Infer.assembly selected)) (fail "Inference load materialization differs from the declared request")
    mapM_ (nonempty fields) (["model", "revision"] ++ ["scope" | Fields.member "scope" fields])
    pure found

consumed :: Object -> Object -> Either String ()
consumed expected fields = unless (Fields.delete "stage" fields == expected) (Left "Inference consumption differs from the declared invocation")

result :: V.Binding -> Object -> Either String ()
result bound = parseEither $ \fields -> do
    Json.fields (["stage", "binding", "adapter", "tokenizer", "base", "assembly", "request", "tokens", "prompt_length", "behavior", "behavior_bits", "text", "truncated"] ++ ["reference" | Fields.member "reference" fields]) fields
    binding bound fields
    fields .: "request" >>= withObject "inference numerical request" (Json.fields ["prompt", "tokens", "temperature", "seed"])

unloaded :: Object -> Either String ()
unloaded = parseEither (Json.fields ["stage", "binding", "program"])

profile :: Object -> Identities -> Either String ()
profile fields found = unless (Fields.lookup "model" fields == Just (String (Text.pack (model found))) && Fields.lookup "revision" fields == Just (String (Text.pack (revision found)))) (Left "Model profile differs from the loaded model or revision")

binding :: V.Binding -> Object -> Parser ()
binding bound fields = do
    actual <- fields .: "binding"
    unless (actual == Wire.bindingValue bound) (fail "Inference observation binding mismatch")

identity :: Object -> Key -> Parser String
identity fields key = fields .: key >>= Json.identity

nonempty :: Object -> Key -> Parser ()
nonempty fields key = do
    value <- fields .: key
    when (Text.null value) (fail "Expected nonempty inference model or scope text")
