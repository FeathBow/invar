{-# LANGUAGE OverloadedStrings #-}

module Invar.Policy.Header (Tensor (..), decode, fp32Bytes) where

import Control.Monad (foldM, unless, when)
import Data.Aeson.Decoding.ByteString (bsToTokens)
import Data.Aeson.Decoding.Tokens
import Data.Aeson.Key qualified as Key
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.List (sortOn)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)

data Field = String Text | Integer Integer | Array [Field] | Object (Map Text Field)

data Tensor = Tensor {name :: Text, shape :: [Integer], begin :: Integer, end :: Integer}
    deriving (Eq, Show)

fp32Bytes :: Integer
fp32Bytes = 4

decode :: ByteString -> Integer -> Either String [Tensor]
decode encoded payloadSize = do
    unless (Bytes.take 1 encoded == "{") (Left "Policy header must begin with an object")
    (parsed, remaining) <- value (bsToTokens encoded)
    unless (Bytes.all (== space) remaining) (Left "Unexpected bytes after policy header")
    fields <- object parsed
    mapM_ metadata (Map.lookup "__metadata__" fields)
    tensors <- traverse tensor (Map.toAscList (Map.delete "__metadata__" fields))
    when (null tensors) (Left "Policy must contain a nonempty tensor inventory")
    occupied <- foldM contiguous 0 (sortOn (\entry -> (begin entry, end entry)) tensors)
    unless (occupied == payloadSize) (Left "Policy tensor offsets do not cover the data buffer")
    pure tensors
  where
    space = 0x20

value :: Tokens rest String -> Either String (Field, rest)
value (TkText text rest) = Right (String text, rest)
value (TkNumber (NumInteger number) rest) = Right (Integer number, rest)
value (TkArrayOpen entries) = array [] entries
value (TkRecordOpen entries) = record Map.empty entries
value (TkErr problem) = Left problem
value _ = Left "Unsupported policy header value"

array :: [Field] -> TkArray rest String -> Either String (Field, rest)
array collected (TkItem next) = do
    (item, remaining) <- value next
    array (item : collected) remaining
array collected (TkArrayEnd rest) = Right (Array (reverse collected), rest)
array _ (TkArrayErr problem) = Left problem

record :: Map Text Field -> TkRecord rest String -> Either String (Field, rest)
record collected (TkPair key next) = do
    let label = Key.toText key
    when (Map.member label collected) (Left "Duplicate policy header key")
    (item, remaining) <- value next
    record (Map.insert label item collected) remaining
record collected (TkRecordEnd rest) = Right (Object collected, rest)
record _ (TkRecordErr problem) = Left problem

object :: Field -> Either String (Map Text Field)
object (Object fields) = Right fields
object _ = Left "Expected a policy header object"

metadata :: Field -> Either String ()
metadata field = do
    entries <- object field
    unless (all text (Map.elems entries)) (Left "Policy metadata must contain only strings")
  where
    text (String _) = True
    text _ = False

tensor :: (Text, Field) -> Either String Tensor
tensor (label, field) = do
    fields <- object field
    unless (Map.keys fields == ["data_offsets", "dtype", "shape"]) (Left "Unexpected policy tensor fields")
    case Map.lookup "dtype" fields of
        Just (String "F32") -> pure ()
        _ -> Left "Policy tensors must use the FP32 adapter profile"
    dimensions <- numbers (Map.lookup "shape" fields)
    offsets <- numbers (Map.lookup "data_offsets" fields)
    case offsets of
        [first, lastByte]
            | lastByte - first == fp32Bytes * product dimensions ->
                Right (Tensor label dimensions first lastByte)
        _ -> Left "Policy tensor shape and byte offsets disagree"

numbers :: Maybe Field -> Either String [Integer]
numbers (Just (Array fields)) = traverse number fields
  where
    number (Integer item) | item >= 0 = Right item
    number _ = Left "Expected nonnegative policy dimensions or offsets"
numbers _ = Left "Expected policy dimensions or byte offsets"

contiguous :: Integer -> Tensor -> Either String Integer
contiguous offset entry = do
    unless (begin entry == offset) (Left "Policy tensor offsets overlap or contain holes")
    pure (end entry)
