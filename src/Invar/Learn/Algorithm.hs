{-# LANGUAGE DataKinds #-}

module Invar.Learn.Algorithm (Sink, Specification, Optimizer (..), sink, specification, coefficients, clipping, penalty, delta, optimizer) where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (symbolVal)

type Sink = "update"

type Specification = "grpo-token-mean/v1"

data Optimizer = Optimizer {learningRate :: Double, firstMoment :: Double, secondMoment :: Double, epsilon :: Double, weightDecay :: Double}
    deriving (Eq, Show)

sink :: String
sink = symbolVal (Proxy :: Proxy Sink)

specification :: String
specification = symbolVal (Proxy :: Proxy Specification)

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
