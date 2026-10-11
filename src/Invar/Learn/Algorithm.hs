{-# LANGUAGE DataKinds #-}

module Invar.Learn.Algorithm (Sink, Grpo, Opd, Optimizer (..), Teacher (..), sink, grpo, opd, coefficients, clipping, penalty, delta, optimizer) where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (symbolVal)

type Sink = "update"

type Grpo = "grpo-token-mean/v1"

type Opd = "opd-token-mean/v1"

data Optimizer = Optimizer {learningRate :: Double, firstMoment :: Double, secondMoment :: Double, epsilon :: Double, weightDecay :: Double}
    deriving (Eq, Show)

data Teacher = Teacher {teacherModel :: String, teacherRevision :: String, teacherTokenizer :: String, teacherBase :: String, teacherAssembly :: String, teacherProfile :: String, teacherTemperature :: Double, teacherTruncation :: String, teacherLogprobs :: String, teacherPath :: String}
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

positive :: Double -> Bool
positive value = finite value && value > 0

nonnegative :: Double -> Bool
nonnegative value = finite value && value >= 0

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)
