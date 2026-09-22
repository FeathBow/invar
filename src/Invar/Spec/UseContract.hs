{-# LANGUAGE Safe #-}

module Invar.Spec.UseContract (
    Budget (..),
    Population (..),
    Standard (..),
    NumericalRequirement (..),
    InvarianceRequirement (..),
    Criterion (..),
    LossRequirement (..),
    Premise (..),
    Reliance (..),
    premiseDescription,
    premiseKind,
) where

import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.List (find)
import Invar.Spec.Numerical qualified as Numerical
import Invar.Spec.Obligation qualified as Obligation
import Numeric.Natural (Natural)

data Budget = Budget {limit :: Rational, rationale :: String}
    deriving (Eq, Show)

data Population = Population
    { populationName :: String
    , unitSampling :: String
    , replicateSampling :: String
    , referenceAlpha :: Rational
    , regressionAlpha :: Rational
    , familyAlpha :: Rational
    }
    deriving (Eq, Show)

data Standard = FiniteDomain | HoeffdingPopulation Population | EmpiricalBernsteinPopulation Population | ConditionalDerivation [Obligation.Obligation]
    deriving (Eq, Show)

data NumericalRequirement = NumericalRequirement
    { relation :: Numerical.Relation
    , probeSteps :: [Natural]
    , numericalRationale :: String
    }
    deriving (Eq, Show)

data LossRequirement = LossRequirement
    { referenceCeiling :: Budget
    , regressionCeiling :: Budget
    , standard :: Standard
    }
    deriving (Eq, Show)

data InvarianceRequirement = InvarianceRequirement
    { invariant :: Numerical.Relation
    , executions :: Natural
    , invarianceRationale :: String
    }
    deriving (Eq, Show)

data Criterion = Criterion
    { numericalRequirements :: [NumericalRequirement]
    , lossRequirement :: Maybe LossRequirement
    , invarianceRequirements :: [InvarianceRequirement]
    }
    deriving (Eq, Show)

-- These are the specific external premises used by existing observation rules,
-- not arbitrary propositions that a caller may whitelist into a proof.
data Premise
    = ParameterMeaning
    | MeasurementMeaning
    | MeasurementInput
    | ExecutionAuthenticity
    | SelectedBehavior
    | OwnCacheExecution
    | ScoreAuthenticity
    | ScoredBehavior
    | OwnCacheScoring
    | ScoringCorrespondence
    | ProbeAuthenticity
    | FullVocabularyBehavior
    | OwnCacheProbing
    | ProbingCorrespondence
    | VocabularyCorrespondence
    | ScheduleVariation
    | IndependentUnits
    | ReplicateSamplingLaw
    | BernsteinIndependentUnits
    | BernsteinIdenticalUnits
    | BernsteinReplicateSamplingLaw
    | RequirementsJustified
    | ContractFrozen
    | AcceptanceIsolation
    | SelectionControl
    deriving (Eq, Ord, Show, Enum, Bounded)

data Reliance = Reliance
    { premise :: Premise
    , authority :: String
    , basis :: ByteString
    }
    deriving (Eq, Show)

premiseDescription :: Premise -> (String, ByteString, String)
premiseDescription selected = case selected of
    ParameterMeaning -> describe "measurement-parameters-valid" "bounded-measurement/v1" "declared-parameter/v1"
    MeasurementMeaning -> describe "measurement-method-meaning" "bounded-measurement/v1" "declared-method/v1"
    MeasurementInput -> describe "measurement-input-correspondence" "bounded-measurement/v1" "bound-inference-field/v1"
    ExecutionAuthenticity -> describe "execution-report-authenticity" "bounded-measurement/v1" "complete-response/v1"
    SelectedBehavior -> generation "selected-behavior-measurement"
    OwnCacheExecution -> generation "own-cache-execution"
    ScoreAuthenticity -> scoring "score-execution-report-authenticity"
    ScoredBehavior -> scoring "scored-behavior-measurement"
    OwnCacheScoring -> scoring "own-cache-scoring"
    ScoringCorrespondence -> scoring "scoring-generation-correspondence"
    ProbeAuthenticity -> probing "probe-execution-report-authenticity"
    FullVocabularyBehavior -> probing "full-vocabulary-behavior-measurement"
    OwnCacheProbing -> probing "own-cache-probing"
    ProbingCorrespondence -> probing "probing-generation-correspondence"
    VocabularyCorrespondence -> probing "vocabulary-coordinate-correspondence"
    ScheduleVariation -> describe "schedule-variation" "finite-paired-inference/v1" "repeated-candidate-execution/v1"
    IndependentUnits -> statistical "independent-unit-sampling"
    ReplicateSamplingLaw -> statistical "declared-replicate-sampling-law"
    BernsteinIndependentUnits -> bernstein "independent-unit-sampling"
    BernsteinIdenticalUnits -> bernstein "identically-distributed-unit-sampling"
    BernsteinReplicateSamplingLaw -> bernstein "declared-replicate-sampling-law"
    RequirementsJustified -> admission "use-requirements-justified"
    ContractFrozen -> admission "candidate-and-contract-frozen-before-acceptance"
    AcceptanceIsolation -> admission "acceptance-data-isolated-from-search"
    SelectionControl -> admission "selection-and-multiplicity-control"
  where
    describe name specification observation = (name, Bytes.pack specification, observation)
    generation name = describe name "finite-paired-inference/v1" "selected-token-behavior/v1"
    scoring name = describe name "cached-path-score/v1" "selected-path-behavior/v1"
    probing name = describe name "cached-distribution-probe/v1" "full-vocabulary-behavior/v1"
    statistical name = describe name "paired-loss-hoeffding/v1" "unit-mean-loss/v1"
    bernstein name = describe name "paired-loss-bernstein-mp2009/v1" "unit-mean-loss/v1"
    admission name = describe name "use-contract/v1" "frozen-declaration/v1"

premiseKind :: Obligation.Obligation -> Maybe Premise
premiseKind obligation
    | matches ("execution-report-authenticity", Bytes.pack "finite-paired-inference/v1", "selected-token-behavior/v1") = Just ExecutionAuthenticity
    | otherwise = find (matches . premiseDescription) [minBound .. maxBound]
  where
    matches (name, specification, observation) =
        takeWhile (/= '/') (Obligation.predicate obligation) == name
            && Obligation.specification obligation == specification
            && Obligation.observation obligation == observation
