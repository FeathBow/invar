{-# LANGUAGE DataKinds #-}

module Invar.Learn.Algorithm (Sink, Grpo, Opd, Optimizer (..), Teacher (..), sink, grpo, opd, coefficients, clipping, penalty, delta, optimizer, teacher) where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (symbolVal)
import Invar.Digest qualified as Digest

type Sink = "update"

type Grpo = "grpo-token-mean/v1"

type Opd = "opd-token-mean/v1"

data Optimizer = Optimizer {learningRate :: Double, firstMoment :: Double, secondMoment :: Double, epsilon :: Double, weightDecay :: Double}
    deriving (Eq, Show)

data Teacher = Teacher {teacherModel :: String, teacherRevision :: String, teacherAdapter :: String, teacherTokenizer :: String, teacherBase :: String, teacherAssembly :: String, teacherTemperature :: Double, teacherTruncation :: String, teacherLogprobs :: String, teacherPath :: String}
    deriving (Eq, Show)

sink :: String
sink = symbolVal (Proxy :: Proxy Sink)

grpo :: String
grpo = symbolVal (Proxy :: Proxy Grpo)

opd :: String
opd = symbolVal (Proxy :: Proxy Opd)

coefficients :: Double -> Double -> Double -> Bool
coefficients chosenClipping chosenPenalty chosenDelta = clipping chosenClipping && penalty chosenPenalty && delta chosenDelta

clipping :: Double -> Bool
clipping value = value > 0 && value < 1

penalty :: Double -> Bool
penalty = nonnegative

delta :: Double -> Bool
delta = positive

optimizer :: Optimizer -> Bool
optimizer chosen = nonnegative (learningRate chosen) && positive (epsilon chosen) && nonnegative (weightDecay chosen) && all moment [firstMoment chosen, secondMoment chosen]
  where
    moment value = value >= 0 && value < 1

teacher :: Teacher -> Either String ()
teacher chosen
    | null (teacherModel chosen) || null (teacherRevision chosen) = Left "The teacher's model and revision must be nonempty"
    | not (all Digest.sha256 [teacherAdapter chosen, teacherTokenizer chosen, teacherBase chosen, teacherAssembly chosen]) = Left "Expected lowercase SHA-256 teacher identities"
    | not (positive (teacherTemperature chosen)) = Left "The teacher's temperature must be positive"
    | (teacherTruncation chosen, teacherLogprobs chosen, teacherPath chosen) /= ("none", "processed", "forced") = Left "The teacher must score the forced path with processed log probabilities and no truncation"
    | otherwise = Right ()

positive :: Double -> Bool
positive value = finite value && value > 0

nonnegative :: Double -> Bool
nonnegative value = finite value && value >= 0

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)
