{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE Safe #-}

module Invar.Spec.Value (
    Scalar (..),
    Value (..),
    Normal,
    normalForm,
) where

import Control.DeepSeq (NFData)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Word (Word32)
import GHC.Generics (Generic)
import Numeric.Natural (Natural)

data Scalar = Boolean Bool | Number Rational | Bits32 Word32 | Token Natural
    deriving (Eq, Show, Generic, NFData)

data Value key
    = Atom Scalar
    | Record (Map String (Value key))
    | Sequence [Value key]
    | Mapping (Map key (Value key))
    deriving (Eq, Show, Generic, NFData)

data Normal
    = Scalar Scalar
    | Fields (Map String Normal)
    | Items [Normal]
    | Entries [(Natural, Normal)]
    deriving (Eq, Show)

type Names key = Map key Natural

normalForm :: (Ord key) => Value key -> Normal
normalForm = snd . normalize Map.empty

normalize :: (Ord key) => Names key -> Value key -> (Names key, Normal)
normalize names value = case value of
    Atom scalar -> (names, Scalar scalar)
    Record fields ->
        let (updated, result) = walk field names (Map.toAscList fields)
         in (updated, Fields (Map.fromDistinctAscList result))
    Sequence values ->
        let (updated, result) = walk normalize names values
         in (updated, Items result)
    Mapping values ->
        let (updated, result) = walk entry names (Map.toAscList values)
         in (updated, Entries result)

field :: (Ord key) => Names key -> (String, Value key) -> (Names key, (String, Normal))
field names (label, value) =
    let (updated, result) = normalize names value
     in (updated, (label, result))

entry :: (Ord key) => Names key -> (key, Value key) -> (Names key, (Natural, Normal))
entry names (key, value) =
    let (extended, index) = name names key
        (updated, result) = normalize extended value
     in (updated, (index, result))

name :: (Ord key) => Names key -> key -> (Names key, Natural)
name names key = case Map.lookup key names of
    Just index -> (names, index)
    Nothing ->
        let index = fromIntegral (Map.size names)
         in (Map.insert key index names, index)

walk :: (state -> input -> (state, output)) -> state -> [input] -> (state, [output])
walk _ state [] = (state, [])
walk visit state (value : rest) =
    let (next, output) = visit state value
        (final, outputs) = walk visit next rest
     in (final, output : outputs)
