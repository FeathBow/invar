{-# LANGUAGE Safe #-}

module Invar.Numerical.Logarithm (Fixed (..), scale, add, times, logarithm) where

import Data.Bits (shiftL)

data Fixed = Fixed !Integer !Integer

precision :: Int
precision = 96

scale :: Integer
scale = 1 `shiftL` precision

terms :: Integer
terms = 32

add :: Fixed -> Fixed -> Fixed
add (Fixed a b) (Fixed c d) = Fixed (a + c) (b + d)

times :: Integer -> Fixed -> Fixed
times factor (Fixed lower upper)
    | factor >= 0 = Fixed (factor * lower) (factor * upper)
    | otherwise = Fixed (factor * upper) (factor * lower)

ceilingQuotient :: Integer -> Integer -> Integer
ceilingQuotient numerator denominator = (numerator + denominator - 1) `div` denominator

multiply :: Fixed -> Fixed -> Fixed
multiply (Fixed a b) (Fixed c d) = Fixed ((a * c) `div` scale) (ceilingQuotient (b * d) scale)

divide :: Fixed -> Integer -> Fixed
divide (Fixed lower upper) divisor = Fixed (lower `div` divisor) (ceilingQuotient upper divisor)

logarithm :: Integer -> Integer -> Fixed
logarithm numerator denominator = reduce numerator denominator 0
  where
    reduce a b power
        | a < b = reduce (2 * a) b (power - 1)
        | a >= 2 * b = reduce a (2 * b) (power + 1)
        | otherwise = add (times power logTwo) (series (a - b) (a + b))

logTwo :: Fixed
logTwo = series 1 3

series :: Integer -> Integer -> Fixed
series 0 _ = Fixed 0 0
series numerator denominator = finish (go 0 z (Fixed 0 0))
  where
    z = Fixed ((scale * numerator) `div` denominator) (ceilingQuotient (scale * numerator) denominator)
    square = multiply z z
    go index power total
        | index == terms = total
        | otherwise = go (index + 1) (multiply power square) (add total (divide power (2 * index + 1)))
    tailBound = ceilingQuotient (scale * 9) (4 * (2 * terms + 1) * 3 ^ (2 * terms + 1))
    finish (Fixed lower upper) = Fixed (2 * lower) (2 * upper + tailBound)
