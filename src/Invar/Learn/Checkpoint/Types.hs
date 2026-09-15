{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Checkpoint.Types (Phase (..), AdamW (..), Expected (..), Check (..), Checked (..), exact, field, mappingValue, sequenceValue, materialization) where

import Control.Monad (unless)
import Data.Aeson.Types (Pair)
import Data.Bifunctor (first)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Word (Word32)
import Invar.Learn.Native qualified as Native

data Phase = Initial | Updated

data AdamW = AdamW {rate :: Double, coefficients :: [Double], epsilon :: Double, decay :: Double}

data Expected = Expected {phase :: Phase, policy :: Text, bindings :: [(Text, Text)], optimizer :: AdamW}

data Check = Finite | Step | IntegerStep Phase | Word32Equal Word32

data Checked = Checked Native.Value [(Check, Native.Tensor)] [Native.Tensor] [Pair]

exact :: (String, [Text]) -> Native.Value -> Either String (Map Text Native.Value)
exact (description, names) encoded = do
    fields <- Native.mapping encoded
    unless (Map.keysSet fields == Set.fromList names) (Left ("Unexpected " ++ description ++ " fields"))
    pure fields

field :: Text -> Map Text Native.Value -> Either String Native.Value
field name = maybe (Left ("Missing checkpoint field: " ++ Text.unpack name)) Right . Map.lookup name

mappingValue :: Map Text Native.Value -> Native.Value
mappingValue = Native.Mapping "builtins.dict" . map (first Native.String) . Map.toAscList

sequenceValue :: [Native.Value] -> Native.Value
sequenceValue = Native.List "builtins.list"

materialization :: Map Text Native.Value -> (Text, Text) -> Either String ()
materialization fields (name, intended) = do
    actual <- field name fields >>= Native.string
    let description = if name == "tokenizer" then "tokenizer" else "model materialization"
    unless (actual == intended) (Left ("Checkpoint " ++ description ++ " differs from the declared input"))
