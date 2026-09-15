{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Native (Value (..), Tensor (..), parse, mapping, indexed, list, tuple, string, integer, number, typeName, tensorValues) where

import Control.Monad (unless, when)
import Data.Aeson ((.:))
import Data.Aeson qualified as Json
import Data.Aeson.Types (Parser)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Word (Word64)
import GHC.Float (castWord64ToDouble)
import Invar.Json qualified as Fields
import Numeric (readHex, showHex)
import Numeric.Natural (Natural)

data Value = Mapping Text [(Value, Value)] | List Text [Value] | Tuple Text [Value] | Tensor Tensor | String Text | Integer Integer | Float Word64 | Boolean Bool | None | Unsupported Text
    deriving (Eq, Show)

data Tensor = Description {index :: Natural, tensorType :: Text, dtype :: Text, shape :: [Integer], size :: Integer, layout :: Text}
    deriving (Eq, Show)

parse :: Json.Value -> Parser Value
parse = Json.withObject "native checkpoint value" $ \fields -> do
    kind <- fields .: "kind" :: Parser Text
    case kind of
        "mapping" -> do
            Fields.fields ["kind", "type", "items"] fields
            Mapping <$> typename fields <*> (fields .: "items" >>= traverse entry)
        "list" -> sequenceValue List fields
        "tuple" -> sequenceValue Tuple fields
        "tensor" -> do
            Fields.fields ["kind", "type", "index", "dtype", "shape", "size", "layout"] fields
            tensor <- Description <$> fields .: "index" <*> typename fields <*> fields .: "dtype" <*> fields .: "shape" <*> fields .: "size" <*> fields .: "layout"
            unless (size tensor >= 0 && all (>= 0) (shape tensor)) (fail "Invalid native tensor dimensions or byte count")
            pure (Tensor tensor)
        "string" -> scalar fields "value" String
        "boolean" -> scalar fields "value" Boolean
        "float" -> scalar fields "bits" Float
        "integer" -> do
            Fields.fields ["kind", "hex"] fields
            text <- fields .: "hex"
            Integer <$> either fail pure (hexadecimal text)
        "none" -> Fields.fields ["kind"] fields >> pure None
        "unsupported" -> scalar fields "type" Unsupported
        _ -> fail "Unknown native checkpoint value kind"
  where
    entry [key, value] = (,) <$> parse key <*> parse value
    entry _ = fail "Expected native mapping key/value entries"
    sequenceValue constructor fields = do
        Fields.fields ["kind", "type", "items"] fields
        constructor <$> typename fields <*> (fields .: "items" >>= traverse parse)
    scalar fields key constructor = Fields.fields ["kind", key] fields >> fmap constructor (fields .: key)
    typename fields = do
        name <- fields .: "type"
        when (Text.null name) (fail "Expected a native Python type name")
        pure name

hexadecimal :: String -> Either String Integer
hexadecimal encoded = do
    let (negative, digits) = case encoded of '-' : rest -> (True, rest); _ -> (False, encoded)
    magnitude <- case readHex digits of
        [(parsed, "")] -> Right parsed
        _ -> Left "Expected a native integral hexadecimal value"
    let decoded = if negative then negate magnitude else magnitude
        canonical = (if decoded < 0 then "-" else "") ++ showHex (abs decoded) ""
    unless (canonical == encoded) (Left "Expected canonical native integer digits")
    pure decoded

mapping :: Value -> Either String (Map Text Value)
mapping (Mapping _ entries) = traverse key entries >>= distinct
  where
    key (String name, value) = Right (name, value)
    key _ = Left "Expected native string mapping keys"
mapping _ = Left "Expected a native mapping"

indexed :: Value -> Either String (Map Integer Value)
indexed (Mapping _ entries) = traverse key entries >>= distinct
  where
    key (Integer name, value) | name >= 0 = Right (name, value)
    key _ = Left "Expected nonnegative native integer mapping keys"
indexed _ = Left "Expected a native parameter mapping"

distinct :: (Ord key) => [(key, value)] -> Either String (Map key value)
distinct entries = do
    let result = Map.fromList entries
    unless (Map.size result == length entries) (Left "Duplicate native mapping keys")
    pure result

list :: Value -> Either String [Value]
list (List _ values) = Right values
list _ = Left "Expected a native list"

tuple :: Value -> Either String [Value]
tuple (Tuple _ values) = Right values
tuple _ = Left "Expected a native tuple"

string :: Value -> Either String Text
string (String value) = Right value
string _ = Left "Expected native text"

integer :: Value -> Either String Integer
integer (Integer value) = Right value
integer _ = Left "Expected a native integer"

number :: Value -> Either String Rational
number (Integer value) = Right (fromInteger value)
number (Float bits) = do
    let value = castWord64ToDouble bits
    when (isNaN value || isInfinite value) (Left "Expected a finite native number")
    pure (toRational value)
number _ = Left "Expected a native integer or floating number"

typeName :: Value -> Text
typeName (Mapping name _) = name
typeName (List name _) = name
typeName (Tuple name _) = name
typeName (Tensor value) = tensorType value
typeName (String _) = "builtins.str"
typeName (Integer _) = "builtins.int"
typeName (Float _) = "builtins.float"
typeName (Boolean _) = "builtins.bool"
typeName None = "builtins.NoneType"
typeName (Unsupported name) = name

tensorValues :: Value -> [Tensor]
tensorValues (Tensor value) = [value]
tensorValues (Mapping _ entries) = concatMap (\(key, value) -> tensorValues key ++ tensorValues value) entries
tensorValues (List _ values) = concatMap tensorValues values
tensorValues (Tuple _ values) = concatMap tensorValues values
tensorValues _ = []
