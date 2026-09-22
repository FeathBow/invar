{-# LANGUAGE Safe #-}

module Invar.Use.Finding (Finding (..), establish, finding, goal) where

import Data.Foldable (toList)
import Data.Map.Strict qualified as Map
import Data.Maybe (mapMaybe)
import Invar.Spec.Evidence qualified as Evidence
import Invar.Spec.Numerical qualified as Numerical
import Invar.Spec.Use qualified as U
import Invar.Spec.UseContract qualified as Contract

data Finding = Finding U.Claim Evidence.Verdict
    deriving (Eq, Show)

establish :: U.Claim -> U.Observed -> Finding
establish target observed = Finding target (Evidence.check graph root)
  where
    root = Evidence.EvidenceId 0
    graph = case target of
        U.Required selected criterion
            | selected == U.scope observed ->
                let targets = requirements selected criterion
                    indexed = zip (map Evidence.EvidenceId [1 ..]) targets
                    dependencies = map fst indexed
                    parent = Evidence.Node (Evidence.All targets) (Evidence.Conjoin dependencies)
                 in Map.fromList ((root, parent) : mapMaybe attach indexed)
        _ -> Map.singleton root (Evidence.Node (Evidence.TaskLoss target) (Evidence.ObserveTaskLoss observed))
    attach (identity, claim) = case claim of
        Evidence.TaskLoss _ -> Just (identity, Evidence.Node claim (Evidence.ObserveTaskLoss observed))
        Evidence.Numerical (Numerical.Claim expected _) -> do
            actual <- Map.lookup (scopeBytes (Numerical.scopeId expected)) numericalByScope
            pure (identity, Evidence.Node claim (Evidence.Observe actual))
        _ -> Nothing
    U.Scope _ _ _ samples = U.scope observed
    -- Full scope equality is still checked by Observe. This map merely locates
    -- the actual observation; it cannot establish equality from a digest.
    numericalByScope = Map.fromList [(scopeBytes (Numerical.scopeId (Numerical.scope actual)), actual) | sample <- toList samples, actual <- U.numerical sample : U.invariance sample]
    scopeBytes (Numerical.ScopeId bytes) = bytes

finding :: Finding -> Evidence.Verdict
finding (Finding _ result) = result

goal :: U.Claim -> Evidence.Claim
goal (U.Required selected criterion) = Evidence.All (requirements selected criterion)
goal target = Evidence.TaskLoss target

requirements :: U.Scope -> Contract.Criterion -> [Evidence.Claim]
requirements selected criterion = case Contract.lossRequirement criterion of
    Just required | Contract.ConditionalDerivation _ <- Contract.standard required -> [Evidence.TaskLoss (U.Required selected criterion)]
    requested -> maybe [] losses requested ++ numerical ++ invariance
  where
    U.Scope _ _ _ samples = selected
    losses requested =
        [ Evidence.TaskLoss (claim (Contract.standard requested) metric (Contract.limit budget))
        | (metric, budget) <- [(U.ReferenceLoss, Contract.referenceCeiling requested), (U.LossIncrease, Contract.regressionCeiling requested)]
        ]
    claim Contract.FiniteDomain metric budget = U.Claim selected metric budget
    claim (Contract.HoeffdingPopulation population) metric budget =
        U.PopulationClaim selected population metric budget
    claim (Contract.EmpiricalBernsteinPopulation population) metric budget =
        U.EmpiricalBernsteinClaim selected population metric budget
    claim (Contract.ConditionalDerivation _) _ _ = U.Required selected criterion
    numerical =
        [ Evidence.Numerical (Numerical.Claim (Numerical.scope (U.numerical sample)) (Contract.relation requirement))
        | sample <- toList samples
        , requirement <- toList (Contract.numericalRequirements criterion)
        ]
    invariance =
        [ Evidence.Numerical (Numerical.Claim (Numerical.scope repeated) (Contract.invariant requirement))
        | sample <- toList samples
        , requirement <- toList (Contract.invarianceRequirements criterion)
        , repeated <- U.invariance sample
        ]
