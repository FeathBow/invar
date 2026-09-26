{-# LANGUAGE Safe #-}

module Invar.Use.Plan (Source (..), Parameter (..), Assumptions (..), Standard (..), Search (..), Bound (..), PlanEstimate (..), estimate, searchLimit, satisfied) where

import Invar.Spec.Use (Metric (..))
import Invar.Spec.UseContract qualified as Contract
import Invar.Use.Confidence qualified as Confidence
import Numeric.Natural (Natural)

data Source = Assumption | Probe {probeUnits :: Natural, probeContract :: String}
    deriving (Eq, Show)

data Parameter = Parameter {parameterValue :: Rational, parameterSource :: Source}
    deriving (Eq, Show)

data Assumptions = Assumptions
    { plannedUnits :: Natural
    , referenceVariance :: Maybe Parameter
    , increaseVariance :: Maybe Parameter
    , referenceMean :: Maybe Parameter
    , increaseMean :: Maybe Parameter
    }
    deriving (Eq, Show)

data Standard = Hoeffding | Bernstein
    deriving (Eq, Show)

data Search = Found Natural | NotFoundWithin Natural | NoFiniteSolution
    deriving (Eq, Show)

data Bound = Bound
    { boundMetric :: Metric
    , boundRange :: Rational
    , boundAlpha :: Rational
    , boundCeiling :: Rational
    , boundVariance :: Maybe Parameter
    , boundMean :: Maybe Parameter
    , boundWidth :: Maybe Rational
    , boundFeasible :: Maybe Bool
    , boundSearch :: Maybe Search
    }
    deriving (Eq, Show)

data PlanEstimate = NotApplicable String String | Estimated Standard Natural [Bound]
    deriving (Eq, Show)

searchLimit :: Natural
searchLimit = 10000000

estimate :: Maybe Contract.LossRequirement -> Assumptions -> PlanEstimate
estimate Nothing _ = NotApplicable "none" "the contract has no loss requirement"
estimate (Just requirement) assumptions = case Contract.standard requirement of
    Contract.FiniteDomain -> NotApplicable "finite_domain" "the decision is the exact mean over the declared units"
    Contract.ConditionalDerivation _ -> NotApplicable "conditional_derivation" "a conditional derivation is always Unknown"
    Contract.HoeffdingPopulation population -> bounds Hoeffding population
    Contract.EmpiricalBernsteinPopulation population -> bounds Bernstein population
  where
    count = plannedUnits assumptions
    bounds standard population =
        Estimated
            standard
            count
            [ bound standard ReferenceLoss 1 (Contract.referenceAlpha population) (Contract.limit (Contract.referenceCeiling requirement)) (referenceVariance assumptions) (referenceMean assumptions)
            , bound standard LossIncrease 2 (Contract.regressionAlpha population) (Contract.limit (Contract.regressionCeiling requirement)) (increaseVariance assumptions) (increaseMean assumptions)
            ]
    bound standard metric range alpha budget variance assumed =
        let width = halfWidth standard range alpha (parameterValue <$> variance)
            feasible = satisfied budget . parameterValue <$> assumed <*> width count
            search = do
                value <- parameterValue <$> assumed
                _ <- width searchLimit
                pure (sizeFor budget value width)
         in Bound metric range alpha budget variance assumed (width count) feasible search

halfWidth :: Standard -> Rational -> Rational -> Maybe Rational -> Natural -> Maybe Rational
halfWidth Hoeffding range alpha _ units = Confidence.hoeffding range alpha units
halfWidth Bernstein range alpha variance units = variance >>= \value -> Confidence.bernsteinWidth range alpha value units

satisfied :: Rational -> Rational -> Rational -> Bool
satisfied budget value width = min 1 (value + width) <= budget

sizeFor :: Rational -> Rational -> (Natural -> Maybe Rational) -> Search
sizeFor budget value width
    | value > budget && budget < 1 = NoFiniteSolution
    | not (holds searchLimit) = NotFoundWithin searchLimit
    | otherwise = Found (smallest 1 searchLimit)
  where
    holds units = maybe False (satisfied budget value) (width units)
    smallest lower upper
        | lower >= upper = upper
        | holds middle = smallest lower middle
        | otherwise = smallest (middle + 1) upper
      where
        middle = (lower + upper) `div` 2
