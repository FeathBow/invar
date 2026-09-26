{-# LANGUAGE OverloadedStrings #-}

module UsePlan (usePlan) where

import Data.Ratio ((%))
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Use qualified as U
import Invar.Use.Confidence qualified as Confidence
import Invar.Use.Plan qualified as Plan

usePlan :: Group
usePlan =
    Group
        "Conditional sample size planning"
        [ ("planning and admission share one width computation", withTests 200 (property shared))
        , ("a finite domain has no population bound to plan", once finite)
        , ("without a mean assumption only the width is reported", once widthOnly)
        , ("a Bernstein width needs a variance", once varianceRequired)
        , ("the smallest sufficient N is exact under the core's comparison", once smallest)
        , ("no finite solution is claimed only when proved", once unsolvable)
        ]
  where
    once = withTests 1 . property

population :: U.Population
population = U.Population "declared pool" "uniform draw" "one seed per unit" (1 % 40) (1 % 40) (1 % 20)

requirement :: U.Standard -> Rational -> Rational -> U.LossRequirement
requirement standard reference regression = U.LossRequirement (U.Budget reference "planning test") (U.Budget regression "planning test") standard

assumed :: Rational -> Maybe Plan.Parameter
assumed value = Just (Plan.Parameter value Plan.Assumption)

assumptions :: Plan.Assumptions
assumptions = Plan.Assumptions 2000 (assumed (1 % 5)) (assumed (1 % 20)) Nothing Nothing

bounds :: Plan.PlanEstimate -> PropertyT IO [Plan.Bound]
bounds (Plan.Estimated _ _ values) = pure values
bounds other = annotateShow other >> failure

increase :: [Plan.Bound] -> PropertyT IO Plan.Bound
increase values = case [value | value <- values, Plan.boundMetric value == U.LossIncrease] of
    [value] -> pure value
    _ -> failure

shared :: PropertyT IO ()
shared = do
    samples <- forAll (Gen.list (Range.linear 2 40) (fmap (% 8) (Gen.integral (Range.linear (-8) 8))))
    variance <- evalMaybe (Confidence.sampleVariance samples)
    Confidence.empiricalBernstein 2 (1 % 40) samples === Confidence.bernsteinWidth 2 (1 % 40) variance (fromIntegral (length samples))

finite :: PropertyT IO ()
finite = case Plan.estimate (Just (requirement U.FiniteDomain (3 % 10) (1 % 20))) assumptions of
    Plan.NotApplicable "finite_domain" _ -> success
    other -> annotateShow other >> failure

widthOnly :: PropertyT IO ()
widthOnly = do
    found <- bounds (Plan.estimate (Just (requirement (U.EmpiricalBernsteinPopulation population) (3 % 10) (1 % 20))) assumptions) >>= increase
    Plan.boundWidth found === Confidence.bernsteinWidth 2 (1 % 40) (1 % 20) 2000
    Plan.boundFeasible found === Nothing
    Plan.boundSearch found === Nothing

varianceRequired :: PropertyT IO ()
varianceRequired = do
    found <- bounds (Plan.estimate (Just (requirement (U.EmpiricalBernsteinPopulation population) (3 % 10) (1 % 20))) assumptions {Plan.increaseVariance = Nothing, Plan.increaseMean = assumed 0}) >>= increase
    Plan.boundWidth found === Nothing
    Plan.boundFeasible found === Nothing
    Plan.boundSearch found === Nothing
    hoeffding <- bounds (Plan.estimate (Just (requirement (U.HoeffdingPopulation population) (3 % 10) (1 % 20))) assumptions {Plan.increaseVariance = Nothing}) >>= increase
    Plan.boundWidth hoeffding === Confidence.hoeffding 2 (1 % 40) 2000

smallest :: PropertyT IO ()
smallest = do
    found <- bounds (Plan.estimate (Just (requirement (U.EmpiricalBernsteinPopulation population) (3 % 10) (1 % 20))) assumptions {Plan.increaseMean = assumed (1 % 100)}) >>= increase
    size <- case Plan.boundSearch found of
        Just (Plan.Found value) -> pure value
        other -> annotateShow other >> failure
    let holds n = maybe False (Plan.satisfied (1 % 20) (1 % 100)) (Confidence.bernsteinWidth 2 (1 % 40) (1 % 20) n)
    assert (holds size)
    assert (not (holds (size - 1)))
    Plan.boundFeasible found === Just (holds 2000)

unsolvable :: PropertyT IO ()
unsolvable = do
    let planned budget mean = bounds (Plan.estimate (Just (requirement (U.EmpiricalBernsteinPopulation population) 1 budget)) assumptions {Plan.increaseMean = assumed mean}) >>= increase
    above <- planned (1 % 20) (1 % 10)
    Plan.boundSearch above === Just Plan.NoFiniteSolution
    atBudget <- planned (1 % 20) (1 % 20)
    Plan.boundSearch atBudget === Just (Plan.NotFoundWithin Plan.searchLimit)
    ceilingOne <- planned 1 2
    Plan.boundSearch ceilingOne === Just (Plan.Found 2)
