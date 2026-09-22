module Main (main) where

import Data.Ratio (denominator, numerator, (%))
import Invar.Use.Confidence (empiricalBernstein, hoeffding)
import Text.Read (readMaybe)

main :: IO ()
main = interact (unlines . map calculate . lines)

calculate :: String -> String
calculate line = case words line of
    "bernstein" : arguments -> maybe "invalid" (render . bernstein) (traverse readMaybe arguments)
    arguments -> maybe "invalid" (render . legacy) (traverse readMaybe arguments)

legacy :: [Integer] -> Maybe Rational
legacy values = case values of
    [rangeNumerator, rangeDenominator, alphaNumerator, alphaDenominator, count]
        | rangeDenominator > 0 && alphaDenominator > 0 && count >= 0 ->
            hoeffding (rangeNumerator % rangeDenominator) (alphaNumerator % alphaDenominator) (fromInteger count)
    _ -> Nothing

-- bernstein rangeNumerator rangeDenominator alphaNumerator alphaDenominator
-- followed by one numerator/denominator pair per independent sampling unit.
bernstein :: [Integer] -> Maybe Rational
bernstein (rangeNumerator : rangeDenominator : alphaNumerator : alphaDenominator : values) = do
    range <- ratio rangeNumerator rangeDenominator
    alpha <- ratio alphaNumerator alphaDenominator
    samples <- ratios values
    empiricalBernstein range alpha samples
bernstein _ = Nothing

ratios :: [Integer] -> Maybe [Rational]
ratios [] = Just []
ratios (valueNumerator : valueDenominator : rest) = (:) <$> ratio valueNumerator valueDenominator <*> ratios rest
ratios _ = Nothing

ratio :: Integer -> Integer -> Maybe Rational
ratio valueNumerator valueDenominator
    | valueDenominator > 0 = Just (valueNumerator % valueDenominator)
    | otherwise = Nothing

render :: Maybe Rational -> String
render (Just value) = show (numerator value) ++ " " ++ show (denominator value)
render Nothing = "invalid"
