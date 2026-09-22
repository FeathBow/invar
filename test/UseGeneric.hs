{-# LANGUAGE OverloadedStrings #-}

module UseGeneric (useGeneric) where

import Data.Aeson (encode)
import Data.ByteString.Lazy qualified as Lazy
import Data.List.NonEmpty (NonEmpty (..))
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Hedgehog
import Invar.Numerical qualified as N
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Evidence qualified as Evidence
import Invar.Spec.Program qualified as P
import Invar.Spec.Value (Scalar (..), Value (..))
import Invar.Use qualified as U
import Invar.Use.Measurement qualified as M
import UseAdmission (contractFor)
import UseFixture qualified as F

useGeneric :: Group
useGeneric =
    Group
        "Generic identified observations"
        [ ("an answer-free event uses the same evidence and admission rules", once event)
        , ("numerical-only contracts require no evaluator or invented losses", once numericalOnly)
        , ("method identity and domain units remain part of the contract", once correspondence)
        , ("measurement outputs cannot escape their declared range or bindings", once invalidMeasurement)
        , ("generic declarations round trip without a workload or standard answer", once encoding)
        ]
  where
    once = withTests 1 . property

eventSpec :: P.Expr -> M.MethodSpec
eventSpec expression = M.MethodSpec bytes (Map.singleton source (M.ObservedField M.Truncated)) "event" "truncation-event/v1" 0 1 M.IncreasingLoss "Reported truncation is a bounded event; fixture only"
  where
    source = P.Semantic "event"
    sources = Map.singleton source P.BooleanType
    sink = P.Sink "truncation-event/v1" P.NumberType (Set.singleton source) Set.empty
    meaning = E.Semantics (P.Schema sources Map.empty (Map.singleton "event" sink)) Map.empty
    bytes = A.encode meaning [P.Emit "event" "truncation-event/v1" expression]

eventMethod :: Either M.Error M.Method
eventMethod = M.prepare (eventSpec (P.If (P.Read (P.Input (P.Semantic "event"))) (number 1) (number 0)))

number :: Rational -> P.Expr
number = P.Constant P.NumberType . Atom . Number

fixture :: PropertyT IO U.BoundRun
fixture = do
    -- F.run constructs bound inference reports; its answer field is unused.
    let trial = F.Trial "input" "Unlabelled input" 7 "unused" "free output" "another free output" False
    before <- F.run 50 N.Reference trial
    after <- F.run 51 N.Candidate trial {F.truncated = True}
    selected <- evalEither eventMethod
    let input = U.Input (U.Key 0 "input") "declared-unit" "Unlabelled input" 2 0.8 7 Map.empty
        domain = U.Domain "unlabelled finite inputs" "fixture/v1" "One declared execution unit; independence is not inferred" (input :| [])
    pure (U.BoundRun domain (Just selected) [U.Case (U.Key 0 "input") (N.BoundRun before after) []])

criterion :: U.Criterion
criterion = U.Criterion [] (Just (U.LossRequirement (U.Budget 0 "Fixture event ceiling") (U.Budget 0 "Fixture regression ceiling") U.FiniteDomain)) []

decision :: U.UseContract -> U.Observed -> U.Decision
decision contract observed = U.admit contract (U.establish (U.Required (U.scope observed) (U.criterion contract)) observed)

event :: PropertyT IO ()
event = do
    supplied <- fixture
    base <- contractFor supplied
    observed <- evalEither (U.observe supplied)
    U.mean U.ReferenceLoss observed === Just 0
    U.mean U.CandidateLoss observed === Just 1
    U.mean U.LossIncrease observed === Just 1
    case decision base {U.criterion = criterion} observed of
        U.Rejected witness -> Evidence.witness witness === Evidence.TaskLossWitness observed
        other -> annotateShow other >> failure
    let permissive = U.Criterion [] (Just (U.LossRequirement (U.Budget 0 "Fixture reference") (U.Budget 1 "Full range fixture only") U.FiniteDomain)) []
    case decision base {U.criterion = permissive} observed of
        U.Admitted admitted -> do
            assert (any ((== U.MeasurementMeaning) . U.premise . U.supporting) (U.conditions admitted))
            assert (all ((/= U.ParameterMeaning) . U.premise . U.supporting) (U.conditions admitted))
        other -> annotateShow other >> failure

numericalOnly :: PropertyT IO ()
numericalOnly = do
    initial <- fixture
    let supplied = initial {U.measurement = Nothing}
    base <- contractFor supplied
    observed <- evalEither (U.observe supplied)
    U.mean U.LossIncrease observed === Nothing
    U.finding (U.establish (U.Claim (U.scope observed) U.LossIncrease 1) observed) === Evidence.Unknown (Evidence.TaskLossProblem U.MissingMeasurement)
    let chosen = base {U.criterion = U.Criterion [U.NumericalRequirement N.SameTokens [] "Fixture token equality"] Nothing []}
    case decision chosen observed of
        U.Admitted admitted -> do
            assert (all ((/= U.MeasurementMeaning) . U.premise . U.supporting) (U.conditions admitted))
            U.declaredMeasurement (U.admissionContract admitted) === Nothing
        other -> annotateShow other >> failure
    case decision base {U.criterion = U.Criterion [] Nothing []} observed of
        U.Undetermined problems -> assert (U.InvalidContract "empty use requirements" `elem` problems)
        other -> annotateShow other >> failure

correspondence :: PropertyT IO ()
correspondence = do
    supplied <- fixture
    base <- contractFor supplied
    observed <- evalEither (U.observe supplied)
    original <- evalEither eventMethod
    other <- evalEither (M.prepare (M.specification original) {M.meaning = "Different declared event meaning"})
    let changed = base {U.criterion = criterion, U.declaredMeasurement = Just other}
    case decision changed observed of
        U.Undetermined problems -> assert (U.MeasurementMismatch `elem` problems)
        result -> annotateShow result >> failure
    let domain = U.domain supplied
        changedDomain = base {U.criterion = criterion, U.declaredDomain = domain {U.unitDefinition = "Another sampling unit definition"}}
    case decision changedDomain observed of
        U.Undetermined problems -> assert (U.DomainMismatch `elem` problems)
        result -> annotateShow result >> failure
    otherDomain <- evalEither (U.observe supplied {U.domain = U.declaredDomain changedDomain})
    let numericalCriterion = U.Criterion [U.NumericalRequirement N.SameTokens [] "fixture"] Nothing []
        mismatched = U.establish (U.Required (U.scope observed) numericalCriterion) otherDomain
    U.finding mismatched === Evidence.Unknown (Evidence.TaskLossProblem (U.ScopeMismatch (U.scopeId (U.scope observed)) (U.scopeId (U.scope otherDomain))))

invalidMeasurement :: PropertyT IO ()
invalidMeasurement = do
    supplied <- fixture
    outside <- evalEither (M.prepare (eventSpec (number 2)))
    case U.observe supplied {U.measurement = Just outside} of
        Left (U.MeasurementFailed _ _ (M.EvaluationFailed (M.OutsideRange 0 1 2))) -> success
        other -> annotateShow other >> failure
    selected <- evalEither eventMethod
    missing <- evalEither (M.prepare (M.specification selected) {M.bindings = Map.singleton (P.Semantic "event") (M.Parameter "missing")})
    case U.observe supplied {U.measurement = Just missing} of
        Left (U.MeasurementFailed _ _ (M.MissingParameter "missing")) -> success
        other -> annotateShow other >> failure
    wrongType <- evalEither (M.prepare (M.specification selected) {M.bindings = Map.singleton (P.Semantic "event") (M.ObservedField M.Response)})
    case U.observe supplied {U.measurement = Just wrongType} of
        Left (U.MeasurementFailed _ _ (M.EvaluationFailed (M.InvalidInputs _))) -> success
        other -> annotateShow other >> failure

encoding :: PropertyT IO ()
encoding = do
    supplied <- fixture
    contract <- contractFor supplied
    let chosen = contract {U.criterion = criterion}
    U.decodeContract (Lazy.toStrict (encode (U.describeContract chosen))) === Right chosen
    let onlyNumeric = chosen {U.declaredMeasurement = Nothing, U.criterion = U.Criterion [U.NumericalRequirement N.SameTokens [] "fixture"] Nothing []}
    U.decodeContract (Lazy.toStrict (encode (U.describeContract onlyNumeric))) === Right onlyNumeric
