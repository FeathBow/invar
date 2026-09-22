{-# LANGUAGE Safe #-}

module Invar.Spec.Use (
    Key (..),
    Sample (..),
    ScopeId (..),
    Scope (..),
    Unit (..),
    Observed (..),
    Metric (..),
    Claim (..),
    Confidence (..),
    Problem (..),
    Judgement (..),
    scope,
    scopeId,
    units,
    mean,
    source,
    judge,
    premises,
    claimScope,
    confidence,
    claimPremises,
    populationValid,
) where

import Data.ByteString (ByteString)
import Data.Char (isSpace)
import Data.Foldable (toList)
import Data.List.NonEmpty (NonEmpty)
import Data.Map.Strict qualified as Map
import Data.Maybe (isNothing)
import Data.Text qualified as Text
import Data.Text.Encoding qualified as Text
import Invar.Spec.Domain (Key (..))
import Invar.Spec.Domain qualified as Domain
import Invar.Spec.Measurement qualified as Measurement
import Invar.Spec.Numerical qualified as Numerical
import Invar.Spec.Obligation (Obligation)
import Invar.Spec.Obligation qualified as Obligation
import Invar.Spec.UseContract qualified as Contract
import Invar.Use.Confidence qualified as Confidence
import Numeric.Natural (Natural)

data Sample = Sample
    { key :: Key
    , unitId :: String
    , numerical :: Numerical.Observed
    , invariance :: [Numerical.Observed]
    , reference :: Maybe Measurement.Measurement
    , candidate :: Maybe Measurement.Measurement
    }
    deriving (Eq, Show)

newtype ScopeId = ScopeId ByteString
    deriving (Eq, Show)

data Scope = Scope ScopeId Domain.Domain (Maybe Measurement.Method) (NonEmpty Sample)
    deriving (Eq, Show)

data Unit = Unit
    { unitName :: String
    , members :: NonEmpty Key
    , referenceLoss :: Maybe Rational
    , candidateLoss :: Maybe Rational
    }
    deriving (Eq, Show)

data Observed = Observed Scope (NonEmpty Unit)
    deriving (Eq, Show)

data Metric = ReferenceLoss | CandidateLoss | LossIncrease
    deriving (Eq, Show)

data Claim
    = Claim Scope Metric Rational
    | PopulationClaim Scope Contract.Population Metric Rational
    | EmpiricalBernsteinClaim Scope Contract.Population Metric Rational
    | Required Scope Contract.Criterion
    deriving (Eq, Show)

data Confidence = Confidence {unitCount :: Natural, alpha :: Rational, width :: Rational, upper :: Rational}
    deriving (Eq, Show)

data Problem
    = ScopeMismatch ScopeId ScopeId
    | InvalidPopulation
    | MissingMeasurement
    | UnsupportedPopulationMetric Metric
    | InsufficientUnits Natural Natural
    | InsufficientLossBound Metric Rational Confidence
    | JointEvidenceRequired
    | UnsupportedDerivation [Obligation]
    deriving (Eq, Show)

data Judgement = Satisfied | Violated | Insufficient Problem
    deriving (Eq, Show)

judge :: Claim -> Observed -> Judgement
judge target observed
    | claimScope target /= scope observed = Insufficient (ScopeMismatch (scopeId (claimScope target)) (scopeId (scope observed)))
    | otherwise = case target of
        Claim _ metric budget -> case mean metric observed of
            Nothing -> Insufficient MissingMeasurement
            Just measured -> if measured <= budget then Satisfied else Violated
        PopulationClaim _ population metric budget -> judgePopulation target observed (population, metric, budget)
        EmpiricalBernsteinClaim _ population metric budget -> judgePopulation target observed (population, metric, budget)
        Required _ criterion -> case Contract.standard <$> Contract.lossRequirement criterion of
            Just (Contract.ConditionalDerivation assumptions) -> Insufficient (UnsupportedDerivation assumptions)
            _ -> Insufficient JointEvidenceRequired

judgePopulation :: Claim -> Observed -> (Contract.Population, Metric, Rational) -> Judgement
judgePopulation target observed (population, metric, budget)
    | not (populationValid population) = Insufficient InvalidPopulation
    | metric == CandidateLoss = Insufficient (UnsupportedPopulationMetric metric)
    | isNothing (mean metric observed) = Insufficient MissingMeasurement
    | count < minimumUnits = Insufficient (InsufficientUnits count minimumUnits)
    | otherwise = case confidence target observed of
        Just result | upper result <= budget -> Satisfied
        Just result -> Insufficient (InsufficientLossBound metric budget result)
        Nothing -> Insufficient InvalidPopulation
  where
    count = fromIntegral (length (units observed))
    minimumUnits = case target of EmpiricalBernsteinClaim {} -> 2; _ -> 1

claimScope :: Claim -> Scope
claimScope (Claim selected _ _) = selected
claimScope (PopulationClaim selected _ _ _) = selected
claimScope (EmpiricalBernsteinClaim selected _ _ _) = selected
claimScope (Required selected _) = selected

populationValid :: Contract.Population -> Bool
populationValid value =
    not (any (all isSpace) [Contract.populationName value, Contract.unitSampling value, Contract.replicateSampling value])
        && all (\a -> a > 0 && a < 1) [Contract.referenceAlpha value, Contract.regressionAlpha value, Contract.familyAlpha value]
        && Contract.referenceAlpha value + Contract.regressionAlpha value <= Contract.familyAlpha value

confidence :: Claim -> Observed -> Maybe Confidence
confidence target observed = do
    (expected, population, metric, _) <- populationClaim target
    if expected /= scope observed || not (populationValid population)
        then Nothing
        else do
            selectedAlpha <- case metric of ReferenceLoss -> Just (Contract.referenceAlpha population); LossIncrease -> Just (Contract.regressionAlpha population); CandidateLoss -> Nothing
            measured <- measurements metric observed
            let count = fromIntegral (length measured)
                range = case metric of LossIncrease -> 2; _ -> 1
            halfWidth <- case target of
                PopulationClaim {} -> Confidence.hoeffding range selectedAlpha count
                EmpiricalBernsteinClaim {} -> Confidence.empiricalBernstein range selectedAlpha (toList measured)
                _ -> Nothing
            pure (Confidence count selectedAlpha halfWidth (min 1 (sum measured / fromIntegral count + halfWidth)))

populationClaim :: Claim -> Maybe (Scope, Contract.Population, Metric, Rational)
populationClaim (PopulationClaim selected population metric budget) = Just (selected, population, metric, budget)
populationClaim (EmpiricalBernsteinClaim selected population metric budget) = Just (selected, population, metric, budget)
populationClaim _ = Nothing

scope :: Observed -> Scope
scope (Observed value _) = value

scopeId :: Scope -> ScopeId
scopeId (Scope value _ _ _) = value

units :: Observed -> NonEmpty Unit
units (Observed _ values) = values

mean :: Metric -> Observed -> Maybe Rational
mean metric observed = do
    measured <- measurements metric observed
    pure (sum measured / fromIntegral (length measured))

measurements :: Metric -> Observed -> Maybe (NonEmpty Rational)
measurements metric observed = traverse measure (units observed)
  where
    measure unit = case metric of
        ReferenceLoss -> referenceLoss unit
        CandidateLoss -> candidateLoss unit
        LossIncrease -> (-) <$> candidateLoss unit <*> referenceLoss unit

source :: Numerical.Side -> Sample -> Numerical.Source
source side sample = case Numerical.scope (numerical sample) of
    Numerical.Scope _ before after _ _ _ -> case side of Numerical.Reference -> before; Numerical.Candidate -> after

premises :: Observed -> [Obligation]
premises observed = methodMeaning ++ parameters ++ responses
  where
    Scope (ScopeId domain) declared selected samples = scope observed
    methodMeaning =
        [ obligation Contract.MeasurementMeaning "" (encoded (Measurement.specification method, Domain.unitDefinition declared))
        | Just method <- [selected]
        ]
    parameters =
        [ obligation
            Contract.ParameterMeaning
            ("/" ++ show (Domain.inputKey input) ++ "/" ++ name)
            (encoded (method, input, value))
        | Just method <- [selected]
        , input <- toList (Domain.declaredInputs declared)
        , Measurement.Parameter name <- Map.elems (Measurement.bindings (Measurement.specification method))
        , Just value <- [Map.lookup name (Domain.parameters input)]
        ]
    responses =
        concat
            [ obligation Contract.ExecutionAuthenticity suffix (encoded (source side sample))
                : [ obligation Contract.MeasurementInput (suffix ++ "/" ++ show field) (encoded (source side sample, Measurement.method measured, Measurement.inputs measured, field))
                  | Measurement.ObservedField field <- Map.elems (Measurement.bindings (Measurement.method measured))
                  ]
            | sample <- toList samples
            , (side, scored) <- [(Numerical.Reference, reference sample), (Numerical.Candidate, candidate sample)]
            , let suffix = "/" ++ show (key sample) ++ "/" ++ show side
            , Just measured <- [scored]
            ]
    obligation kind suffix =
        let (name, specification, observation) = Contract.premiseDescription kind
         in Obligation.Obligation (name ++ suffix) specification observation domain

claimPremises :: Claim -> Observed -> [Obligation]
claimPremises target observed =
    premises observed ++ case populationClaim target of
        Just (_, population, metric, budget) ->
            [ Obligation.Obligation name specification observation domain (encoded (population, metric, budget))
            | kind <- sampling ++ [Contract.ContractFrozen, Contract.AcceptanceIsolation, Contract.SelectionControl]
            , let (name, specification, observation) = Contract.premiseDescription kind
            ]
        _ -> []
  where
    ScopeId domain = scopeId (scope observed)
    sampling = case target of
        EmpiricalBernsteinClaim {} -> [Contract.BernsteinIndependentUnits, Contract.BernsteinIdenticalUnits, Contract.BernsteinReplicateSamplingLaw]
        _ -> [Contract.IndependentUnits, Contract.ReplicateSamplingLaw]

encoded :: (Show value) => value -> ByteString
encoded = Text.encodeUtf8 . Text.pack . show
