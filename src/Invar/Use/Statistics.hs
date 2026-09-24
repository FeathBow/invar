{-# LANGUAGE Safe #-}

module Invar.Use.Statistics (Statistics (..), McNemar (..), compare, normalQuantileUpper, normalLower) where

import Data.List (genericLength)
import Data.Ratio ((%))
import Invar.Use.Confidence (empiricalBernstein, hoeffding)
import Prelude hiding (compare)

data McNemar = McNemar {worse :: Integer, better :: Integer, oneSided :: Rational}
    deriving (Eq, Show)

data Statistics = Statistics
    { units :: Integer
    , meanIncrease :: Rational
    , hoeffdingUpper :: Rational
    , bernsteinUpper :: Rational
    , waldLower :: Rational
    , waldUpper :: Rational
    , equivalent :: Bool
    , noninferior :: Bool
    , mcnemar :: Maybe McNemar
    }
    deriving (Eq, Show)

compare :: Rational -> Rational -> [(Rational, Rational)] -> Maybe Statistics
compare alpha margin pairs = do
    let increases = [candidate - reference | (reference, candidate) <- pairs]
        count = genericLength increases
        average = sum increases / fromInteger count
    hoeffdingWidth <- hoeffding 2 alpha (fromInteger count)
    bernsteinWidth <- empiricalBernstein 2 alpha increases
    quantile <- normalQuantileUpper alpha
    variance <- if count >= 2 then Just (sum [(value - average) ^ (2 :: Int) | value <- increases] / fromInteger (count - 1)) else Nothing
    let waldWidth = quantile * rootUpper (variance / fromInteger count)
        lower = average - waldWidth
        upper = average + waldWidth
    pure
        Statistics
            { units = count
            , meanIncrease = average
            , hoeffdingUpper = average + hoeffdingWidth
            , bernsteinUpper = average + bernsteinWidth
            , waldLower = lower
            , waldUpper = upper
            , equivalent = lower > negate margin && upper < margin
            , noninferior = upper < margin
            , mcnemar = exact pairs
            }

exact :: [(Rational, Rational)] -> Maybe McNemar
exact pairs
    | all (\(reference, candidate) -> binary reference && binary candidate) pairs = Just (McNemar worseCount betterCount tail')
    | otherwise = Nothing
  where
    binary value = value == 0 || value == 1
    worseCount = genericLength [() | (0, 1) <- pairs]
    betterCount = genericLength [() | (1, 0) <- pairs]
    discordant = worseCount + betterCount
    tail' = sum [choose discordant k | k <- [worseCount .. discordant]] % (2 ^ discordant)

choose :: Integer -> Integer -> Integer
choose n k = product [n - k + 1 .. n] `div` product [1 .. k]

normalQuantileUpper :: Rational -> Maybe Rational
normalQuantileUpper alpha
    | alpha <= 0 || alpha >= 1 / 2 || normalLower limit < 1 - alpha = Nothing
    | otherwise = Just (search 0 limit)
  where
    limit = 8
    search low high
        | high - low <= 1 % (2 ^ (48 :: Int)) = high
        | normalLower middle >= 1 - alpha = search low middle
        | otherwise = search middle high
      where
        middle = (low + high) / 2

normalLower :: Rational -> Rational
normalLower x
    | integral <= 0 = 1 / 2
    | otherwise = 1 / 2 + integral / rootUpper (2 * piUpper)
  where
    terms = [x ^ (2 * k + 1) / fromInteger (2 ^ k * product [1 .. k] * (2 * k + 1)) | k <- [0 ..]]
    count = until (\k -> fromInteger k > x * x && terms !! fromInteger k < 1 % (10 ^ (40 :: Int))) (+ 1) (1 :: Integer)
    partial = sum [(if even k then id else negate) (terms !! fromInteger k) | k <- [0 .. count - 1]]
    integral = partial - terms !! fromInteger count

piUpper :: Rational
piUpper = 3141592653589794 % 1000000000000000

rootUpper :: Rational -> Rational
rootUpper value = ceilingRoot (ceiling (value * fromInteger (scale * scale))) % scale
  where
    scale = 10 ^ (30 :: Int)

ceilingRoot :: Integer -> Integer
ceilingRoot value = search 0 (value + 1)
  where
    search lower upper
        | upper - lower <= 1 = if lower * lower == value then lower else upper
        | midpoint * midpoint >= value = search lower midpoint
        | otherwise = search midpoint upper
      where
        midpoint = (lower + upper) `div` 2
