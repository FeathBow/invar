{-# LANGUAGE Safe #-}

module Invar.Reward.Decimal (parse, whitespace, lineBreaks, decimalBase) where

import Data.Char (isDigit)
import Data.List (dropWhileEnd, stripPrefix)
import Data.Ratio ((%))

parse :: String -> Maybe Rational
parse text = stripPrefix "#### " (strip text) >>= signed

signed :: String -> Maybe Rational
signed ('-' : rest) = negate <$> unsigned rest
signed ('+' : rest) = unsigned rest
signed text = unsigned text

unsigned :: String -> Maybe Rational
unsigned text = case span isDigit text of
    ([], _) -> Nothing
    (whole, []) -> Just (fromInteger (integer whole))
    (whole, '.' : fraction)
        | not (null fraction) && all isDigit fraction ->
            let scale = decimalBase ^ length fraction
             in Just ((integer whole * scale + integer fraction) % scale)
    _ -> Nothing

decimalBase :: Integer
decimalBase = 10

integer :: String -> Integer
integer = foldl' (\value character -> decimalBase * value + fromIntegral (fromEnum character - fromEnum '0')) 0

strip :: String -> String
strip = dropWhile (`elem` whitespace) . dropWhileEnd (`elem` whitespace)

whitespace :: String
whitespace = "\t\n\v\f\r\x1c\x1d\x1e\x1f \x85\xa0\x1680\x2028\x2029\x202f\x205f\x3000" ++ ['\x2000' .. '\x200a']

lineBreaks :: String
lineBreaks = "\n\r\v\f\x1c\x1d\x1e\x85\x2028\x2029"
