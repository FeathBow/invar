{-# LANGUAGE OverloadedStrings #-}

module Invar.Use (
    Key (..),
    Input (..),
    Domain (..),
    Case (..),
    BoundRun (..),
    ObservationError (..),
    Scope,
    ScopeId,
    Observed,
    Unit (..),
    Metric (..),
    Claim (..),
    Confidence (..),
    Problem (..),
    Budget (..),
    Population (..),
    Standard (..),
    NumericalRequirement (..),
    InvarianceRequirement (..),
    Criterion (..),
    LossRequirement (..),
    Premise (..),
    Reliance (..),
    UseContract (..),
    Decision (..),
    Admission,
    AdmissionProblem (..),
    ReliedOn (..),
    Finding,
    observe,
    establish,
    finding,
    admit,
    admissionContract,
    admissionScope,
    evidence,
    conditions,
    confidence,
    scope,
    scopeId,
    units,
    mean,
    numerical,
    describe,
    describeFinding,
    describeDecision,
    decodeContract,
    describeContract,
    describeConfidence,
    bounds,
    lossClaims,
) where

import Data.Aeson (Value, object, (.=))
import Data.List.NonEmpty (NonEmpty)
import Data.Ratio (denominator, numerator)
import Invar.Artifact qualified as Artifact
import Invar.Evidence.Encoding qualified as EvidenceEncoding
import Invar.Numerical qualified as Numerical
import Invar.Spec.Domain (Domain (..), Input (..))
import Invar.Spec.Evidence qualified as Evidence
import Invar.Spec.Measurement qualified as Measurement
import Invar.Spec.Use (Claim (..), Confidence (..), Key (..), Metric (..), Observed, Problem (..), Scope, ScopeId, Unit (..), confidence, mean, scope, scopeId, units)
import Invar.Spec.Use qualified as U
import Invar.Use.Admission (Admission, AdmissionProblem (..), Decision (..), ReliedOn (..), admissionContract, admissionScope, admit, conditions, evidence)
import Invar.Use.Contract (Budget (..), Criterion (..), InvarianceRequirement (..), LossRequirement (..), NumericalRequirement (..), Population (..), Premise (..), Reliance (..), Standard (..), UseContract (..))
import Invar.Use.Contract qualified as Contract
import Invar.Use.Encoding (decodeContract, describeContract)
import Invar.Use.Encoding qualified as Encoding
import Invar.Use.Finding (Finding (..), establish, finding, lossClaims)
import Invar.Use.Observation (BoundRun (..), Case (..), ObservationError (..), observe)

describeFinding :: Finding -> Value
describeFinding (Finding target result) =
    object
        ( [ "scope_sha256" .= Artifact.hex identity
          , "use_admission" .= ("not_evaluated" :: String)
          , "judgement" .= EvidenceEncoding.verdict witnessValue result
          ]
            ++ description
        )
  where
    U.ScopeId identity = scopeId (U.claimScope target)
    description = case target of
        Claim _ metric budget -> ["metric" .= show metric, "budget" .= rational budget, "strength" .= ("finite_domain" :: String)]
        PopulationClaim _ population metric budget -> ["metric" .= show metric, "budget" .= rational budget, "strength" .= ("population_hoeffding" :: String), "population" .= Encoding.standardValue (HoeffdingPopulation population)]
        EmpiricalBernsteinClaim _ population metric budget -> ["metric" .= show metric, "budget" .= rational budget, "strength" .= ("population_bernstein_mp2009" :: String), "population" .= Encoding.standardValue (EmpiricalBernsteinPopulation population)]
        Required _ selected -> ["requirements" .= Encoding.criterionValue selected]

witnessValue :: Evidence.Witness -> Value
witnessValue (Evidence.TaskLossWitness observed) = describe observed
witnessValue (Evidence.NumericalWitness observed) = Numerical.describe observed
witnessValue (Evidence.OutputWitness _ _) = object ["kind" .= ("output_bytes" :: String)]

describeDecision :: Decision -> Value
describeDecision (Undetermined reasons) = object ["status" .= ("unknown" :: String), "reasons" .= map show reasons]
describeDecision (Rejected counterexample) = object ["status" .= ("observed_violation" :: String), "finding" .= EvidenceEncoding.verdict witnessValue (Evidence.Refute counterexample)]
describeDecision (Admitted admitted) =
    object
        [ "status" .= ("admitted_under_declared_reliance" :: String)
        , "purpose" .= purpose (admissionContract admitted)
        , "scope_sha256" .= Artifact.hex identity
        , "requirements" .= Encoding.criterionValue (criterion (admissionContract admitted))
        , "methods" .= map Evidence.methodName (Evidence.methods (evidence admitted))
        , "remaining_conditions" .= map condition (conditions admitted)
        ]
  where
    U.ScopeId identity = scopeId (admissionScope admitted)
    condition (ReliedOn external selected) = object ["obligation" .= EvidenceEncoding.premise (Evidence.External external), "authority" .= Contract.authority selected, "basis_sha256" .= Artifact.hex (Contract.basis selected)]

bounds :: Finding -> [(Metric, Confidence)]
bounds = Evidence.bounds . finding

describeConfidence :: Confidence -> Value
describeConfidence bound = object ["units" .= unitCount bound, "alpha" .= rational (alpha bound), "width" .= rational (width bound), "upper" .= rational (upper bound)]

numerical :: Observed -> NonEmpty (Key, Numerical.Observed)
numerical observed = fmap (\sample -> (U.key sample, U.numerical sample)) samples
  where
    U.Scope _ _ _ samples = scope observed

describe :: Observed -> Value
describe observed =
    object
        [ "kind" .= ("finite-paired-observation/v1" :: String)
        , "scope" .= Artifact.hex identity
        , "domain" .= Encoding.domainValue document
        , "measurement" .= fmap Encoding.methodValue measured
        , "sample_count" .= length samples
        , "unit_count" .= length (units observed)
        , "weighting" .= ("equal declared units; equal members within each unit; repeated candidate executions add neither samples nor units" :: String)
        , "reference_loss" .= fmap rational (mean ReferenceLoss observed)
        , "candidate_loss" .= fmap rational (mean CandidateLoss observed)
        , "loss_increase" .= fmap rational (mean LossIncrease observed)
        , "units" .= fmap unitValue (units observed)
        , "samples" .= fmap sampleValue samples
        , "unestablished" .= (["measurement meaning and parameter validity", "independent sampling", "schedule variation between repeated executions", "population guarantee", "use admission"] :: [String])
        ]
  where
    U.Scope (U.ScopeId identity) document measured samples = scope observed

unitValue :: Unit -> Value
unitValue unit = object ["unit" .= unitName unit, "members" .= fmap keyValue (members unit), "reference_loss" .= fmap rational (referenceLoss unit), "candidate_loss" .= fmap rational (candidateLoss unit)]

sampleValue :: U.Sample -> Value
sampleValue sample = object ["key" .= keyValue (U.key sample), "unit" .= U.unitId sample, "reference_measurement" .= fmap measurementValue (U.reference sample), "candidate_measurement" .= fmap measurementValue (U.candidate sample), "numerical" .= Numerical.describe (U.numerical sample), "invariance" .= map Numerical.describe (U.invariance sample)]

measurementValue :: Measurement.Measurement -> Value
measurementValue measured = object ["raw" .= rational (Measurement.raw measured), "normalized_loss" .= rational (Measurement.value measured), "inputs" .= show (Measurement.inputs measured), "emission" .= show (Measurement.emission measured)]

keyValue :: Key -> Value
keyValue selected = object ["cohort" .= cohort selected, "task" .= task selected]

rational :: Rational -> Value
rational value = object ["numerator" .= numerator value, "denominator" .= denominator value]
