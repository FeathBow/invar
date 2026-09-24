{-# LANGUAGE Safe #-}

module Invar.Use.Statistics (Statistics (..), McNemar (..), Bound (..), compare, normalQuantileUpper, normalLower) where

import Data.List (genericLength)
import Data.Ratio ((%))
import Invar.Use.Confidence (empiricalBernstein, hoeffding, sampleVariance, squareRootUpper)
import Prelude hiding (compare)

data Bound = Hoeffding | EmpiricalBernstein
    deriving (Eq, Show)

data McNemar = McNemar {worse :: Integer, better :: Integer, oneSided :: Rational}
    deriving (Eq, Show)

data Statistics = Statistics
    { units :: Integer
    , meanIncrease :: Rational
    , alternative :: (Bound, Rational)
    , waldLower :: Rational
    , waldUpper :: Rational
    , equivalent :: Bool
    , noninferior :: Bool
    , mcnemar :: Maybe McNemar
    }
    deriving (Eq, Show)

compare :: Rational -> Rational -> Bound -> [(Rational, Rational)] -> Maybe Statistics
compare alpha margin used pairs = do
    let increases = [candidate - reference | (reference, candidate) <- pairs]
        count = genericLength increases
        average = sum increases / fromInteger count
    quantile <- normalQuantileUpper alpha
    variance <- sampleVariance increases
    other <- case used of
        EmpiricalBernstein -> (,) Hoeffding . (average +) <$> hoeffding 2 alpha (fromInteger count)
        Hoeffding -> (,) EmpiricalBernstein . (average +) <$> empiricalBernstein 2 alpha increases
    let waldWidth = quantile * squareRootUpper (variance / fromInteger count)
        lower = average - waldWidth
        upper = average + waldWidth
    pure
        Statistics
            { units = count
            , meanIncrease = average
            , alternative = other
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
    | otherwise = 1 / 2 + integral / squareRootUpper (2 * piUpper)
  where
    terms = [x ^ (2 * k + 1) / fromInteger (2 ^ k * product [1 .. k] * (2 * k + 1)) | k <- [0 ..]]
    count = until (\k -> fromInteger k > x * x && terms !! fromInteger k < 1 % (10 ^ (40 :: Int))) (+ 1) (1 :: Integer)
    partial = sum [(if even k then id else negate) (terms !! fromInteger k) | k <- [0 .. count - 1]]
    integral = partial - terms !! fromInteger count

piUpper :: Rational
piUpper = 3141592653589794 % 1000000000000000
