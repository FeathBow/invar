{-# LANGUAGE Safe #-}

module Invar.Use.Finding (Finding (..), establish, finding, goal, goalClaim, lossClaims) where

import Data.Foldable (toList)
import Data.Map.Strict qualified as Map
import Data.Maybe (mapMaybe)
import Invar.Paths qualified as Paths
import Invar.Spec.Evidence qualified as Evidence
import Invar.Spec.Numerical qualified as Numerical
import Invar.Spec.Use qualified as U
import Invar.Spec.UseContract qualified as Contract

data Finding = Finding U.Claim Evidence.Claim Evidence.Verdict U.Observed
    deriving (Eq, Show)

establish :: U.Claim -> U.Observed -> Finding
establish target observed = Finding target (goalClaim target observed) (Evidence.check graph root) observed
  where
    root = Evidence.EvidenceId 0
    graph = case target of
        U.Required selected criterion
            | selected == U.scope observed ->
                let targets = requirements selected criterion attachments
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
        Evidence.Paths (Paths.ScopedClaim expected _) -> do
            actual <- Map.lookup expected pathByScope
            pure (identity, Evidence.Node claim (Evidence.ObservePaths actual))
        _ -> Nothing
    U.Scope _ _ _ samples = U.scope observed
    U.Observed _ _ attachments = observed
    numericalByScope = Map.fromList [(scopeBytes (Numerical.scopeId (Numerical.scope actual)), actual) | sample <- toList samples, actual <- U.numerical sample : U.invariance sample]
    pathByScope = Map.fromList [(Paths.scope actual, actual) | (_, observations) <- attachments, actual <- observations]
    scopeBytes (Numerical.ScopeId bytes) = bytes

finding :: Finding -> Evidence.Verdict
finding (Finding _ _ result _) = result

goal :: Finding -> Evidence.Claim
goal (Finding _ expected _ _) = expected

goalClaim :: U.Claim -> U.Observed -> Evidence.Claim
goalClaim (U.Required selected criterion) observed = Evidence.All (requirements selected criterion attachments)
  where
    U.Observed _ _ attachments = observed
goalClaim target _ = Evidence.TaskLoss target

requirements :: U.Scope -> Contract.Criterion -> [(U.Key, [Paths.Observed])] -> [Evidence.Claim]
requirements selected criterion attachments = case Contract.lossRequirement criterion of
    Just required | Contract.ConditionalDerivation _ <- Contract.standard required -> [Evidence.TaskLoss (U.Required selected criterion)]
    _ -> map Evidence.TaskLoss (lossClaims selected criterion) ++ concatMap numerical samples ++ invariance
  where
    U.Scope _ _ _ samples = selected
    numerical sample = concatMap (forSample sample) (Contract.numericalRequirements criterion)
    forSample sample requirement = case (Numerical.pathRelation (Contract.relation requirement), Numerical.pathSide (Contract.relation requirement)) of
        (Just relation, Just _) ->
            let candidates = [actual | (key, observations) <- attachments, key == U.key sample, actual <- observations]
             in case candidates of
                    [] -> [Evidence.Paths (Paths.ScopedClaim (Paths.missingScope (U.key sample) relation) relation)]
                    _ -> [Evidence.Paths (Paths.ScopedClaim (Paths.scope actual) relation) | actual <- candidates]
        _ -> [Evidence.Numerical (Numerical.Claim (Numerical.scope (U.numerical sample)) (Contract.relation requirement))]
    invariance =
        [ Evidence.Numerical (Numerical.Claim (Numerical.scope repeated) (Contract.invariant requirement))
        | sample <- toList samples
        , requirement <- toList (Contract.invarianceRequirements criterion)
        , repeated <- U.invariance sample
        ]

lossClaims :: U.Scope -> Contract.Criterion -> [U.Claim]
lossClaims selected criterion = case Contract.lossRequirement criterion of
    Just requested ->
        [ claim (Contract.standard requested) metric (Contract.limit budget)
        | (metric, budget) <- [(U.ReferenceLoss, Contract.referenceCeiling requested), (U.LossIncrease, Contract.regressionCeiling requested)]
        ]
    Nothing -> []
  where
    claim Contract.FiniteDomain metric budget = U.Claim selected metric budget
    claim (Contract.HoeffdingPopulation population) metric budget = U.PopulationClaim selected population metric budget
    claim (Contract.EmpiricalBernsteinPopulation population) metric budget = U.EmpiricalBernsteinClaim selected population metric budget
    claim (Contract.ConditionalDerivation _) _ _ = U.Required selected criterion
