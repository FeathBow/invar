{-# LANGUAGE OverloadedStrings #-}

module Invar.Replay.Load (admit, unloaded) where

import Control.Monad (unless, void, when)
import Data.Aeson (Object, Value (..), object, toJSON, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text (Text)
import Data.Text qualified as Text
import Invar.Infer.Model qualified as Model
import Invar.Json qualified as Json
import Invar.Materialization qualified as Materialization
import Invar.Replay.Call qualified as Call
import Invar.Spec.Load qualified as Load

admit :: Call.Call -> Object -> Parser Object
admit observed fields = do
    selected <- Model.binding fields
    expected <- Model.binding (Call.consumed observed)
    let current = Fields.member "load" fields || Fields.member "image" fields
        required = ["stage", "binding", "requested", "consumed", "model", "revision"] ++ Model.fields selected
    Json.fields (required ++ ["scope" | Fields.member "scope" fields] ++ [key | current, key <- ["load", "image"]]) fields
    unless (selected == expected) (fail "Loaded materialization differs from the result")
    requested <- fields .: "requested" >>= Json.identity
    consumed <- fields .: "consumed" >>= Json.identity
    unless (requested == consumed && Just (toJSON consumed) == Fields.lookup "adapter" (Call.consumed observed) && Fields.lookup "binding" fields == Fields.lookup "binding" (Call.consumed observed)) (fail "Loaded model binding differs from the result")
    model <- text "model"
    revision <- text "revision"
    when (Fields.member "scope" fields) (void (text "scope"))
    when current $ do
        actualLoad <- fields .: "load"
        unless (Just actualLoad == Fields.lookup "load" (Call.consumed observed)) (fail "Loaded invocation differs from consumption")
        expectedImage <- case selected of
            Model.Materialized tokenizer base assembly -> pure (Materialization.image (consumed, tokenizer, base, assembly))
            _ -> fail "Current load image requires complete materialization"
        actualImage <- fields .: "image"
        unless (actualImage == object ["artifact" .= Bytes.unpack (Load.artifact expectedImage), "profile" .= Bytes.unpack (Load.profile expectedImage)]) (fail "Loaded image differs from materialization")
    pure (Fields.union (Fields.fromList ["model" .= model, "revision" .= revision]) (case Model.value selected of Object value -> value; _ -> Fields.empty))
  where
    text key = do
        value <- fields .: key :: Parser Text
        when (Text.null value) (fail "Missing model identity or scope")
        pure value

unloaded :: Call.Call -> Object -> Parser ()
unloaded previous fields = do
    Json.fields ["stage", "binding", "program"] fields
    expected <- maybe (fail "Unloaded record has no preceding load invocation") pure (Fields.lookup "load" (Call.consumed previous))
    unless (Object (Fields.delete "stage" fields) == expected) (fail "Unloaded invocation differs from the preceding load")
