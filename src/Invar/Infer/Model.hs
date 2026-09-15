{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Model (Model (..), binding, fields, value) where

import Control.Monad (unless)
import Data.Aeson (Object, Value, object, (.:), (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser)
import Invar.Json qualified as Json

data Model = Unbound | Tokenizer String | Materialized String String String
    deriving (Eq, Show)

binding :: Object -> Parser Model
binding original
    | Fields.member "base" original || Fields.member "assembly" original = do
        unless (all (`Fields.member` original) ["tokenizer", "base", "assembly"]) (fail "Incomplete model materialization binding: tokenizer, base and assembly are required")
        Materialized <$> identity "tokenizer" <*> identity "base" <*> identity "assembly"
    | Fields.member "tokenizer" original = Tokenizer <$> identity "tokenizer"
    | otherwise = pure Unbound
  where
    identity key = original .: key >>= Json.identity

fields :: Model -> [Key]
fields Unbound = []
fields (Tokenizer _) = ["tokenizer"]
fields (Materialized {}) = ["tokenizer", "base", "assembly"]

value :: Model -> Value
value Unbound = object []
value (Tokenizer tokenizer) = object ["tokenizer" .= tokenizer]
value (Materialized tokenizer base assembly) = object ["tokenizer" .= tokenizer, "base" .= base, "assembly" .= assembly]
