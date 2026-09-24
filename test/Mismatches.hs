{-# LANGUAGE OverloadedStrings #-}

module Mismatches (mismatches) where

import Data.Aeson (object, (.=))
import Data.Either (isLeft)
import Data.Word (Word32)
import GHC.Float (castFloatToWord32, castWord32ToFloat, float2Double)
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Learn.Mismatch qualified as M

mismatches :: Group
mismatches = Group "Learner and engine gap" [("known words give exact counts, quantiles and means", withTests 1 (property known)), ("generated words match an independent reference", withTests 200 (property generated)), ("empty and uneven observations are rejected", withTests 1 (property rejected))]

sample :: [Word32] -> [Word32] -> ([Word32], [Word32])
sample = (,)

words32 :: [Float] -> [Word32]
words32 = map castFloatToWord32

known :: PropertyT IO ()
known = do
    summary <- evalEither (M.summarize [sample (words32 [-1, -2, -0.5]) (words32 [-1, -1.5, -0.75]), sample (words32 [-3]) (words32 [-3.25])])
    summary === M.Summary {M.tokens = 4, M.identical = 1, M.mean = 0, M.meanAbsolute = 1 / 4, M.quantiles = [("p50", 1 / 4), ("p90", 1 / 2), ("p99", 1 / 2)], M.largest = 1 / 2}
    M.describe summary === object ["tokens" .= (4 :: Int), "identical" .= (1 :: Int), "gap" .= object ["p50" .= (0.25 :: Double), "p90" .= (0.5 :: Double), "p99" .= (0.5 :: Double), "mean" .= (0 :: Double), "mean_absolute" .= (0.25 :: Double), "max" .= (0.5 :: Double)]]

generated :: PropertyT IO ()
generated = do
    behavior <- forAll (Gen.list (Range.linear 1 64) logProbability)
    proximal <- forAll (traverse (\word -> Gen.choice [pure word, logProbability]) behavior)
    summary <- evalEither (M.summarize [sample behavior proximal])
    let gaps = [exact p - exact b | (b, p) <- zip behavior proximal]
        sizes = map abs gaps
        values = map snd (M.quantiles summary)
    M.tokens summary === length behavior
    M.identical summary === length (filter id (zipWith (==) behavior proximal))
    M.largest summary === maximum sizes
    M.mean summary === sum gaps / fromIntegral (length gaps)
    assert (and (zipWith (<=) values (drop 1 values ++ [M.largest summary])))
    assert (all (`elem` sizes) values)
    assert (M.meanAbsolute summary >= abs (M.mean summary))
  where
    exact = toRational . float2Double . castWord32ToFloat
    logProbability = castFloatToWord32 . negate <$> Gen.float (Range.exponentialFloat 0 30)

rejected :: PropertyT IO ()
rejected = do
    assert (isLeft (M.summarize []))
    assert (isLeft (M.summarize [sample [] []]))
    assert (isLeft (M.summarize [sample (words32 [-1, -2]) (words32 [-1])]))
