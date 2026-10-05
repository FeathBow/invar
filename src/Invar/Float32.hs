{-# LANGUAGE Safe #-}

module Invar.Float32 (finite, logProbability, double, widened) where

import Data.Word (Word32, Word64)
import GHC.Float (castDoubleToWord64, castWord32ToFloat, float2Double)

finite :: Word32 -> Bool
finite word = let value = castWord32ToFloat word in not (isNaN value || isInfinite value)

logProbability :: Word32 -> Bool
logProbability word = finite word && castWord32ToFloat word <= 0

double :: Word32 -> Double
double = float2Double . castWord32ToFloat

widened :: Word32 -> Word64
widened = castDoubleToWord64 . double
