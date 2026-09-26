{-# LANGUAGE Safe #-}

module Invar.Use.Confidence (hoeffding, empiricalBernstein, bernsteinWidth, squareRootUpper, sampleVariance) where

import Data.List (genericLength)
import Data.Ratio (denominator, numerator, (%))
import Invar.Numerical.Logarithm (Fixed (..), logarithm, scale)
import Numeric.Natural (Natural)

hoeffding :: Rational -> Rational -> Natural -> Maybe Rational
hoeffding range alpha count
    | range <= 0 || alpha <= 0 || alpha >= 1 || count == 0 = Nothing
    | otherwise = Just (squareRootUpper (range * range * (upper % scale) / (2 * fromIntegral count)))
  where
    Fixed _ upper = logarithm (denominator alpha) (numerator alpha)

empiricalBernstein :: Rational -> Rational -> [Rational] -> Maybe Rational
empiricalBernstein range alpha samples = do
    variance <- sampleVariance samples
    if maximum samples - minimum samples > range
        then Nothing
        else bernsteinWidth range alpha variance (genericLength samples)

bernsteinWidth :: Rational -> Rational -> Rational -> Natural -> Maybe Rational
bernsteinWidth range alpha variance units
    | range <= 0 || alpha <= 0 || alpha >= 1 || variance < 0 || units < 2 = Nothing
    | otherwise = Just (squareRootUpper (2 * variance * logUpper / count) + correction)
  where
    count = fromIntegral units
    Fixed _ upper = logarithm (2 * denominator alpha) (numerator alpha)
    logUpper = upper % scale
    correction = 7 * range * logUpper / (3 * (count - 1))

sampleVariance :: [Rational] -> Maybe Rational
sampleVariance samples@(_ : _ : _) = Just (sum (map squaredDeviation samples) / (count - 1))
  where
    count = genericLength samples
    mean = sum samples / count
    squaredDeviation value = (value - mean) * (value - mean)
sampleVariance _ = Nothing

squareRootUpper :: Rational -> Rational
squareRootUpper value = ceilingRoot (ceiling (value * fromInteger (scale * scale))) % scale

ceilingRoot :: Integer -> Integer
ceilingRoot value = search 0 (value + 1)
  where
    search lower upper
        | upper - lower <= 1 = if lower * lower == value then lower else upper
        | midpoint * midpoint >= value = search lower midpoint
        | otherwise = search midpoint upper
      where
        midpoint = (lower + upper) `div` 2
