{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE Safe #-}

module Invar.Use.Admission (
    Decision (..),
    Admission,
    AdmissionProblem (..),
    ReliedOn (..),
    admit,
    admissionContract,
    admissionScope,
    evidence,
    conditions,
) where

import Crypto.Hash.SHA256 qualified as SHA256
import Data.ByteString qualified as Bytes
import Data.Char (isSpace)
import Data.Either (partitionEithers)
import Data.Foldable (toList)
import Data.List (find, nub)
import Data.Maybe (isJust, isNothing)
import Data.Text qualified as Text
import Data.Text.Encoding qualified as Text
import Invar.Infer qualified as Infer
import Invar.Policy.Description qualified as Policy
import Invar.Spec.Evidence qualified as Evidence
import Invar.Spec.Numerical qualified as Numerical
import Invar.Spec.Use qualified as U
import Invar.Use.Contract qualified as C
import Invar.Use.Finding (Finding (..), goal)
import Numeric.Natural (Natural)

data Decision = Admitted Admission | Rejected Evidence.Counterexample | Undetermined [AdmissionProblem]
    deriving (Eq, Show)

data Admission = Admission C.UseContract U.Scope Evidence.Certificate [ReliedOn]
    deriving (Eq, Show)

data ReliedOn = ReliedOn {obligation :: Evidence.Obligation, supporting :: C.Reliance}
    deriving (Eq, Show)

data AdmissionProblem
    = InvalidContract String
    | RequirementsMismatch
    | DomainMismatch
    | MeasurementMismatch
    | ImplementationMismatch Numerical.Side U.Key
    | InsufficientExecutions U.Key Natural Natural
    | ContextExceeded U.Key
    | ProbeDomainMismatch U.Key Numerical.Side
    | FindingUnknown Evidence.Problem
    | FindingConclusionMismatch
    | AssumedEvidence
    | UnresolvedClaim Evidence.Claim
    | UnsupportedPremise Evidence.Obligation
    | MissingReliance Evidence.Obligation
    | TransferredViolation
    deriving (Eq, Show)

admit :: C.UseContract -> Finding -> Decision
admit contract (Finding target result)
    | not (null problems) = Undetermined problems
    | otherwise = case result of
        Evidence.Unknown reason -> Undetermined [FindingUnknown reason]
        Evidence.Refute counterexample
            | transferred contract target -> Undetermined [TransferredViolation]
            | otherwise -> Rejected counterexample
        Evidence.Accept certificate
            | Evidence.conclusion certificate /= goal target -> Undetermined [FindingConclusionMismatch]
            | Evidence.Hypothesis `elem` Evidence.methods certificate -> Undetermined [AssumedEvidence]
            | otherwise ->
                let (unresolved, remaining) = partitionEithers (map (supported contract) (Evidence.assumptions certificate ++ map Evidence.External (contractPremises contract selected)))
                 in if null unresolved then Admitted (Admission contract selected certificate remaining) else Undetermined unresolved
  where
    selected = U.claimScope target
    problems = validate contract ++ matches contract target

admissionContract :: Admission -> C.UseContract
admissionContract (Admission contract _ _ _) = contract

admissionScope :: Admission -> U.Scope
admissionScope (Admission _ selected _ _) = selected

evidence :: Admission -> Evidence.Certificate
evidence (Admission _ _ certificate _) = certificate

conditions :: Admission -> [ReliedOn]
conditions (Admission _ _ _ remaining) = remaining

meaningful :: String -> Bool
meaningful = not . all isSpace

validate :: C.UseContract -> [AdmissionProblem]
validate contract = fields ++ budgets ++ numerical ++ invariance ++ standard ++ reliance ++ transfers
  where
    criterion = C.criterion contract
    fields =
        [InvalidContract name | (name, value) <- [("purpose", C.purpose contract), ("freeze protocol", C.freezeProtocol contract), ("isolation protocol", C.isolationProtocol contract), ("selection protocol", C.selectionProtocol contract)], not (meaningful value)]
            ++ [InvalidContract "maximum context" | C.maximumContext contract == 0]
            ++ [InvalidContract "empty use requirements" | null (C.numericalRequirements criterion) && isNothing (C.lossRequirement criterion)]
            ++ [InvalidContract "loss requirements need an identified measurement" | isJust (C.lossRequirement criterion) && isNothing (C.declaredMeasurement contract)]
    budgets = [InvalidContract name | Just requested <- [C.lossRequirement criterion], (name, lower, budget) <- [("reference loss budget", 0, C.referenceCeiling requested), ("loss increase budget", -1, C.regressionCeiling requested)], C.limit budget < lower || C.limit budget > 1 || not (meaningful (C.rationale budget))]
    numerical = concatMap validNumerical (C.numericalRequirements criterion)
    invariance = concatMap validInvariance (C.invarianceRequirements criterion)
    standard = [InvalidContract "population and joint alpha allocation" | Just value <- [population], not (U.populationValid value)]
    population = case C.standard <$> C.lossRequirement criterion of
        Just (C.HoeffdingPopulation value) -> Just value
        Just (C.EmpiricalBernsteinPopulation value) -> Just value
        _ -> Nothing
    declared = C.reliance contract
    reliance =
        [InvalidContract "duplicate premise reliance" | length (nub (map C.premise declared)) /= length declared]
            ++ [InvalidContract ("reliance for " ++ show (C.premise value)) | value <- declared, not (meaningful (C.authority value)) || Bytes.length (C.basis value) /= digestBytes]
    digestBytes = 32
    transfers =
        [InvalidContract "one transfer per side" | length (nub (map C.transferSide (C.transfers contract))) /= length (C.transfers contract)]
            ++ concatMap (validTransfer contract) (C.transfers contract)

validTransfer :: C.UseContract -> C.Transfer -> [AdmissionProblem]
validTransfer contract transfer =
    [InvalidContract "transfer preservation" | not (meaningful (C.preservation transfer))]
        ++ [InvalidContract "transfer between identical implementations" | previous == current]
        ++ [InvalidContract "transfer beyond the implementation assembly" | previous {Policy.assembly = Policy.assembly current} /= current]
  where
    previous = C.previous transfer
    current = C.implementation contract (C.transferSide transfer)

validNumerical :: C.NumericalRequirement -> [AdmissionProblem]
validNumerical requirement = [InvalidContract "numerical requirement rationale" | not (meaningful (C.numericalRationale requirement))] ++ steps
  where
    selected = C.probeSteps requirement
    steps = case C.relation requirement of
        Numerical.FullVocabularyKLWithin {} -> [InvalidContract "frozen probe steps" | null selected || not (and (zipWith (<) selected (drop 1 selected)))]
        _ -> [InvalidContract "probe steps on a non-distribution requirement" | not (null selected)]

validInvariance :: C.InvarianceRequirement -> [AdmissionProblem]
validInvariance requirement =
    [InvalidContract "invariance rationale" | not (meaningful (C.invarianceRationale requirement))]
        ++ [InvalidContract "invariance executions" | C.executions requirement < 2]
        ++ [InvalidContract "invariance relation" | C.invariant requirement `notElem` [Numerical.SameBehaviorBits, Numerical.SameTokens, Numerical.SameTermination]]

matches :: C.UseContract -> U.Claim -> [AdmissionProblem]
matches contract target = requested ++ domain ++ method ++ implementations ++ repeated ++ executions ++ contexts ++ probes
  where
    U.Scope _ declared measured samples = U.claimScope target
    requested = case target of U.Required _ criterion | criterion == C.criterion contract -> []; _ -> [RequirementsMismatch]
    domain = [DomainMismatch | declared /= C.declaredDomain contract]
    method = [MeasurementMismatch | measured /= C.declaredMeasurement contract]
    implementations =
        [ ImplementationMismatch side (U.key sample)
        | sample <- toList samples
        , side <- [Numerical.Reference, Numerical.Candidate]
        , not (admissible contract side (Numerical.sourcePolicy (U.source side sample)))
        ]
    repeated =
        [ ImplementationMismatch Numerical.Candidate (U.key sample)
        | sample <- toList samples
        , observed <- U.invariance sample
        , let Numerical.Scope _ previous next _ _ _ = Numerical.scope observed
        , not (all (admissible contract Numerical.Candidate . Numerical.sourcePolicy) [previous, next])
        ]
    executions =
        [ InsufficientExecutions (U.key sample) required actual
        | sample <- toList samples
        , requirement <- C.invarianceRequirements (C.criterion contract)
        , let required = C.executions requirement
              actual = fromIntegral (length (U.invariance sample)) + 1
        , actual < required
        ]
    contexts =
        [ ContextExceeded (U.key sample)
        | sample <- toList samples
        , let Numerical.Scope _ reference _ prefix _ _ = Numerical.scope (U.numerical sample)
        , fromIntegral (length prefix) + Infer.tokens (Numerical.sourceRequest reference) > C.maximumContext contract
        ]
    probes =
        [ ProbeDomainMismatch (U.key sample) side
        | sample <- toList samples
        , requirement <- toList (C.numericalRequirements (C.criterion contract))
        , Numerical.FullVocabularyKLWithin side _ _ <- [C.relation requirement]
        , Just distributions <- [lookup side (Numerical.distributions (U.numerical sample))]
        , map Numerical.step distributions /= C.probeSteps requirement
        ]

supported :: C.UseContract -> Evidence.Claim -> Either AdmissionProblem ReliedOn
supported contract (Evidence.External premise) = case C.premiseKind premise of
    Nothing -> Left (UnsupportedPremise premise)
    Just kind -> case find ((== kind) . C.premise) (C.reliance contract) of
        Nothing -> Left (MissingReliance premise)
        Just justification -> Right (ReliedOn premise justification)
supported _ premise = Left (UnresolvedClaim premise)

transferred :: C.UseContract -> U.Claim -> Bool
transferred contract target =
    or
        [ Numerical.sourcePolicy source /= C.implementation contract side
        | sample <- toList samples
        , (side, source) <-
            [(Numerical.Reference, U.source Numerical.Reference sample), (Numerical.Candidate, U.source Numerical.Candidate sample)]
                ++ [(Numerical.Candidate, source) | observed <- U.invariance sample, let Numerical.Scope _ previous next _ _ _ = Numerical.scope observed, source <- [previous, next]]
        ]
  where
    U.Scope _ _ _ samples = U.claimScope target

admissible :: C.UseContract -> Numerical.Side -> Policy.Description -> Bool
admissible contract side actual = actual == C.implementation contract side || any declared (C.transfers contract)
  where
    declared transfer = C.transferSide transfer == side && C.previous transfer == actual

contractPremises :: C.UseContract -> U.Scope -> [Evidence.Obligation]
contractPremises contract selected =
    [ Evidence.Obligation name specification observation domain binding
    | kind <- [C.RequirementsJustified, C.ContractFrozen, C.AcceptanceIsolation, C.SelectionControl]
    , let (name, specification, observation) = C.premiseDescription kind
    ]
        ++ [ Evidence.Obligation (name ++ "/" ++ show (U.key sample) ++ "/" ++ show index) specification observation domain (digest (previous, next))
           | not (null (C.invarianceRequirements (C.criterion contract)))
           , sample <- toList samples
           , (index, observed) <- zip [0 :: Natural ..] (U.invariance sample)
           , let Numerical.Scope _ previous next _ _ _ = Numerical.scope observed
                 (name, specification, observation) = C.premiseDescription C.ScheduleVariation
           ]
        ++ [ Evidence.Obligation (name ++ "/" ++ show (C.transferSide transfer)) specification observation domain (digest (transfer, C.implementation contract (C.transferSide transfer), C.criterion contract, C.declaredDomain contract, C.declaredMeasurement contract))
           | let (name, specification, observation) = C.premiseDescription C.ImplementationPreservation
           , transfer <- C.transfers contract
           ]
  where
    U.Scope (U.ScopeId domain) _ _ samples = selected
    binding = digest contract
    digest :: (Show value) => value -> Bytes.ByteString
    digest = SHA256.hash . Text.encodeUtf8 . Text.pack . show
