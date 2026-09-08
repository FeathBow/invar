module Invar.Json (decode) where

import Control.Monad (when)
import Data.Aeson (Value, eitherDecodeStrict)
import Data.Aeson.Decoding.ByteString (bsToTokens)
import Data.Aeson.Decoding.Tokens
import Data.Aeson.Key (Key)
import Data.ByteString (ByteString)
import Data.Set (Set)
import Data.Set qualified as Set

decode :: ByteString -> Either String Value
decode encoded = do
    _ <- value (bsToTokens encoded)
    eitherDecodeStrict encoded

value :: Tokens rest String -> Either String rest
value (TkLit _ rest) = Right rest
value (TkText _ rest) = Right rest
value (TkNumber _ rest) = Right rest
value (TkArrayOpen entries) = array entries
value (TkRecordOpen entries) = record Set.empty entries
value (TkErr problem) = Left problem

array :: TkArray rest String -> Either String rest
array (TkItem next) = value next >>= array
array (TkArrayEnd rest) = Right rest
array (TkArrayErr problem) = Left problem

record :: Set Key -> TkRecord rest String -> Either String rest
record seen (TkPair key next) = do
    when (Set.member key seen) (Left "Duplicate JSON key")
    remaining <- value next
    record (Set.insert key seen) remaining
record _ (TkRecordEnd rest) = Right rest
record _ (TkRecordErr problem) = Left problem
