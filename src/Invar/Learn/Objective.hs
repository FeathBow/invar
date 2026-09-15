module Invar.Learn.Objective (Profile (..), Inputs (..), Output (..), Error (..), reference, calculate, mean32) where

import Control.Monad (foldM, unless, when)
import Data.Word (Word32)
import GHC.Float (castFloatToWord32, castWord32ToFloat, double2Float, float2Double)

data Profile = Profile {epsilon :: Double, penalty :: Double}
    deriving (Eq, Show)

data Inputs = Inputs {current :: Word32, proximal :: Word32, behavior :: Word32, fixed :: Word32, advantage :: Word32}
    deriving (Eq, Show)

data Output = Output {term :: Word32, gradient :: Word32, rewardGradient :: Word32}
    deriving (Eq, Show)

data Error = InvalidInput String | NonFinite String
    deriving (Eq, Show)

data Constants = Constants {lower :: Float, upper :: Float, coefficient :: Float, count :: Float}

reference :: String
reference = "grpo-scalar-f32/v1"

calculate :: Profile -> Int -> [Inputs] -> Either Error [Output]
calculate profile total inputs = do
    unless (total > 0 && not (null inputs) && length inputs <= total) (Left (InvalidInput "Invalid active token count"))
    unless (finite (epsilon profile) && epsilon profile > 0 && epsilon profile < 1) (Left (InvalidInput "Invalid clipping epsilon"))
    unless (finite (penalty profile) && penalty profile >= 0) (Left (InvalidInput "Invalid reference penalty"))
    constants <- Constants <$> rounded "lower clip" (1 - epsilon profile) <*> rounded "upper clip" (1 + epsilon profile) <*> rounded "penalty coefficient" (penalty profile) <*> rounded "active token count" (fromIntegral total)
    traverse (token constants) inputs

token :: Constants -> Inputs -> Either Error Output
token constants inputs = do
    p <- probability (current inputs)
    old <- probability (proximal inputs)
    b <- probability (behavior inputs)
    q <- probability (fixed inputs)
    a <- operand (advantage inputs)
    weight <- binary ("importance difference", (-)) old b >>= exponential
    ratio <- binary ("current difference", (-)) p old >>= exponential
    difference <- binary ("reference difference", (-)) q p
    referenceRatio <- exponential difference
    (selected, slope) <- surrogate constants ratio a
    reward <- binary ("reward term", (*)) (-weight) selected
    distance <- binary ("reference ratio minus difference", (-)) referenceRatio difference >>= \value -> binary ("reference distance", (-)) value 1
    regularizer <- binary ("reference penalty", (*)) (coefficient constants) distance
    value <- binary ("objective term", (+)) reward regularizer
    currentSlope <- binary ("surrogate current slope", (*)) slope ratio
    rewardSlope <- binary ("reward slope", (*)) (-weight) currentSlope
    referenceSlope <- binary ("reference slope", (-)) 1 referenceRatio >>= binary ("penalty slope", (*)) (coefficient constants)
    objectiveSlope <- binary ("objective slope", (+)) rewardSlope referenceSlope
    normalized <- binary ("normalized objective slope", (/)) objectiveSlope (count constants)
    normalizedReward <- binary ("normalized reward slope", (/)) rewardSlope (count constants)
    pure (Output (castFloatToWord32 value) (castFloatToWord32 normalized) (castFloatToWord32 normalizedReward))

surrogate :: Constants -> Float -> Float -> Either Error (Float, Float)
surrogate constants ratio a = do
    direct <- binary ("direct surrogate", (*)) ratio a
    bounded <- binary ("bounded surrogate", (*)) clipped a
    pure $ if direct <= bounded then (direct, a) else (bounded, if interior then a else 0)
  where
    interior = lower constants < ratio && ratio < upper constants
    clipped = min (upper constants) (max (lower constants) ratio)

mean32 :: [Word32] -> Either Error Word32
mean32 words32 = do
    when (null words32) (Left (InvalidInput "Empty objective mean"))
    values <- traverse operand words32
    total <- foldM (binary ("token sum", (+))) 0 values
    divisor <- rounded "active token count" (fromIntegral (length words32))
    castFloatToWord32 <$> binary ("token mean", (/)) total divisor

probability :: Word32 -> Either Error Float
probability encoded = do
    value <- operand encoded
    unless (value <= 0) (Left (InvalidInput "Log probabilities must be nonpositive"))
    pure value

operand :: Word32 -> Either Error Float
operand encoded =
    let value = castWord32ToFloat encoded
     in if finite value then Right value else Left (NonFinite "FP32 operand")

exponential :: Float -> Either Error Float
exponential value = do
    result <- rounded "probability ratio" (exp (float2Double value))
    unless (result > 0) (Left (InvalidInput "Probability ratio underflow"))
    pure result

binary :: (String, Double -> Double -> Double) -> Float -> Float -> Either Error Float
binary (stage, operation) left right = rounded stage (operation (float2Double left) (float2Double right))

rounded :: String -> Double -> Either Error Float
rounded stage value =
    let result = double2Float value
     in if finite value && finite result then Right result else Left (NonFinite stage)

finite :: (RealFloat number) => number -> Bool
finite value = not (isNaN value || isInfinite value)
