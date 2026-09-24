{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Mismatch (Summary (..), summarize, describe) where

import Data.Aeson (Value, object, (.=))
import Data.Aeson.Key qualified as Key
import Data.List (sort)
import Data.Word (Word32)
import GHC.Float (castWord32ToFloat, float2Double)

data Summary = Summary
    { tokens :: Int
    , identical :: Int
    , mean :: Rational
    , meanAbsolute :: Rational
    , quantiles :: [(String, Rational)]
    , largest :: Rational
    }
    deriving (Eq, Show)

summarize :: [([Word32], [Word32])] -> Either String Summary
summarize samples = do
    pairs <- concat <$> traverse paired samples
    if null pairs then Left "A learner and engine comparison requires at least one token" else pure (fromPairs pairs)
  where
    paired (behavior, proximal) = if length behavior == length proximal then Right (zip behavior proximal) else Left "Behavior and proximal token counts differ"

fromPairs :: [(Word32, Word32)] -> Summary
fromPairs pairs =
    Summary
        { tokens = count
        , identical = length (filter (uncurry (==)) pairs)
        , mean = sum gaps / fromIntegral count
        , meanAbsolute = sum sizes / fromIntegral count
        , quantiles = [(name, rank level) | (name, level) <- [("p50", 1 / 2), ("p90", 9 / 10), ("p99", 99 / 100)]]
        , largest = last sizes
        }
  where
    count = length pairs
    gaps = [exact proximal - exact behavior | (behavior, proximal) <- pairs]
    sizes = sort (map abs gaps)
    rank :: Rational -> Rational
    rank level = sizes !! (ceiling (level * fromIntegral count) - 1)
    exact = toRational . float2Double . castWord32ToFloat

describe :: Summary -> Value
describe summary =
    object
        [ "tokens" .= tokens summary
        , "identical" .= identical summary
        , "gap" .= object ([Key.fromString name .= decimal value | (name, value) <- quantiles summary] ++ ["mean" .= decimal (mean summary), "mean_absolute" .= decimal (meanAbsolute summary), "max" .= decimal (largest summary)])
        ]
  where
    decimal value = fromRational value :: Double
