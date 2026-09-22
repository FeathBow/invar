{-# LANGUAGE Safe #-}

module Invar.Numerical.Distribution (Bounds (..), enclose) where

import Control.Monad (foldM, unless)
import Data.Bits (shiftL, shiftR, (.&.))
import Data.Ratio ((%))
import Data.Word (Word32)
import Invar.Numerical.Logarithm (Fixed (..), add, logarithm, scale, times)

data Bounds = FiniteBounds Rational Rational | InfiniteKL
    deriving (Eq, Show)

enclose :: [Word32] -> [Word32] -> Either String (Bounds, Bounds)
enclose left right = do
    unless (not (null left) && length left == length right) (Left "KL vectors must have the same nonempty vocabulary")
    totalP <- total left
    totalQ <- total right
    unless (totalP > 0 && totalQ > 0) (Left "KL vectors must each have positive total mass")
    if and (zipWith (\a b -> magnitude a * totalQ == magnitude b * totalP) left right)
        then pure (FiniteBounds 0 0, FiniteBounds 0 0)
        else pure (finish (totalP, totalQ) (foldl' accumulate empty (zip left right)))

-- Validate every word before a proportional or zero-support shortcut. Strict
-- totals avoid retaining expanded Integer vectors alongside the input words.
total :: [Word32] -> Either String Integer
total = foldM addMass 0
  where
    addMass previous encoded = do
        value <- mass encoded
        pure $! previous + value

data Reduction = Reduction !Fixed !Fixed !Bool !Bool

empty :: Reduction
empty = Reduction (Fixed 0 0) (Fixed 0 0) False False

accumulate :: Reduction -> (Word32, Word32) -> Reduction
accumulate (Reduction p q forward backward) (left, right)
    | a == 0 || b == 0 = Reduction p q (forward || a > 0 && b == 0) (backward || b > 0 && a == 0)
    | otherwise = let value = logarithm a b in Reduction (add p (times a value)) (add q (times (-b) value)) forward backward
  where
    a = magnitude left
    b = magnitude right

finish :: (Integer, Integer) -> Reduction -> (Bounds, Bounds)
finish (totalP, totalQ) (Reduction sumP sumQ forwardInfinite backwardInfinite) = (forward, backward)
  where
    normalization = logarithm totalQ totalP
    forward = if forwardInfinite then InfiniteKL else bounds totalP (add sumP (times totalP normalization))
    backward = if backwardInfinite then InfiniteKL else bounds totalQ (add sumQ (times (-totalQ) normalization))
    bounds size (Fixed lower upper) = FiniteBounds (lower % (size * scale)) (upper % (size * scale))

-- Every nonnegative FP32 mass is an exact integer multiple of 2^-149.
mass :: Word32 -> Either String Integer
mass encoded
    | encoded > 0x3f800000 && encoded /= 0x80000000 = Left "Expected a finite FP32 mass in [0,1]"
    | otherwise = pure (magnitude encoded)

-- Decode validated words again without retaining an expanded mass vector.
magnitude :: Word32 -> Integer
magnitude encoded
    | encoded == 0x80000000 = 0
    | encodedExponent == 0 = mantissa
    | otherwise = (hiddenBit + mantissa) `shiftL` (encodedExponent - 1)
  where
    encodedExponent = fromIntegral (encoded `shiftR` 23)
    mantissa = toInteger (encoded .&. 0x7fffff)
    hiddenBit = 1 `shiftL` 23
