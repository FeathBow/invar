{-# LANGUAGE OverloadedStrings #-}

module UseEvidence (useEvidence) where

import Calls (field)
import Data.Aeson (Value (..))
import Data.ByteString qualified as Bytes
import Data.List (isPrefixOf, nub)
import Data.List.NonEmpty qualified as NonEmpty
import Data.Map.Strict qualified as Map
import Data.Ratio ((%))
import Hedgehog
import Invar.Numerical qualified as N
import Invar.Spec.Evidence qualified as E
import Invar.Use qualified as U
import UseFixture qualified as F

useEvidence :: Group
useEvidence =
    Group
        "Finite task-loss evidence"
        [ ("exact finite means support boundaries and signed improvement claims", once finite)
        , ("measured violations retain the complete task-loss witness", once violation)
        , ("same score under a different answer scope is not interchangeable", once mismatch)
        , ("all grading and response premises remain individually bound", once premises)
        , ("task and numerical evidence compose only through declared graph nodes", once composition)
        , ("assuming a goal or premise leaves that assumption unresolved", once assumptions)
        ]
  where
    once = withTests 1 . property

fixture :: PropertyT IO U.Observed
fixture = F.fixture F.trials >>= evalEither . U.observe

target :: U.Observed -> U.Claim
target observed = U.Claim (U.scope observed) U.LossIncrease (1 % 6)

accepted :: E.Verdict -> PropertyT IO E.Certificate
accepted (E.Accept certificate) = pure certificate
accepted other = annotateShow other >> failure

finite :: PropertyT IO ()
finite = do
    observed <- fixture
    let found = U.establish (target observed) observed
    certificate <- accepted (U.finding found)
    E.conclusion certificate === E.TaskLoss (target observed)
    E.methods certificate === [E.TaskLossObservation]
    field "strength" (U.describeFinding found) === String "finite_domain"
    field "use_admission" (U.describeFinding found) === String "not_evaluated"
    improved <- F.fixture [trial {F.before = F.after trial, F.after = F.before trial} | trial <- F.trials] >>= evalEither . U.observe
    _ <- accepted (U.finding (U.establish (U.Claim (U.scope improved) U.LossIncrease ((-1) % 6)) improved))
    pure ()

violation :: PropertyT IO ()
violation = do
    observed <- fixture
    let claim = U.Claim (U.scope observed) U.LossIncrease 0
    case U.finding (U.establish claim observed) of
        E.Refute counterexample -> do
            E.refuted counterexample === E.TaskLoss claim
            E.witness counterexample === E.TaskLossWitness observed
        other -> annotateShow other >> failure

mismatch :: PropertyT IO ()
mismatch = do
    observed <- fixture
    other <- F.fixture [trial {F.answer = "#### 99"} | trial <- F.trials] >>= evalEither . U.observe
    let problem = E.TaskLossProblem (U.ScopeMismatch (U.scopeId (U.scope observed)) (U.scopeId (U.scope other)))
    U.finding (U.establish (target observed) other) === E.Unknown problem
    E.check (Map.singleton (E.EvidenceId 0) (E.Node (E.TaskLoss (target observed)) (E.ObserveTaskLoss other))) (E.EvidenceId 0) === E.Unknown problem

premises :: PropertyT IO ()
premises = do
    observed <- fixture
    certificate <- accepted (U.finding (U.establish (target observed) observed))
    let remaining = E.assumptions certificate
        obligations = [value | E.External value <- remaining]
    length obligations === 29
    length (nub remaining) === 29
    length (filter (isPrefixOf "measurement-parameters-valid/" . E.predicate) obligations) === 4
    length (filter ((== "complete-response/v1") . E.observation) obligations) === 8
    length (filter ((== "bound-inference-field/v1") . E.observation) obligations) === 16
    length (nub (map E.domain obligations)) === 1
    assert (not (any (Bytes.null . E.binding) obligations))

composition :: PropertyT IO ()
composition = do
    observed <- fixture
    let measured = snd (NonEmpty.head (U.numerical observed))
        numeric = E.Numerical (N.Claim (N.scope measured) N.SameBehaviorBits)
        loss = E.TaskLoss (target observed)
        a = E.EvidenceId 0
        b = E.EvidenceId 1
        c = E.EvidenceId 2
        graph = Map.fromList [(a, E.Node loss (E.ObserveTaskLoss observed)), (b, E.Node numeric (E.Observe measured))]
    certificate <- accepted (E.check (Map.insert c (E.Node (E.All [loss, numeric]) (E.Conjoin [a, b])) graph) c)
    length (E.assumptions certificate) === 35
    E.check (Map.insert c (E.Node loss (E.Conjoin [a, b])) graph) c === E.Unknown (E.WrongConclusion (E.All [loss, numeric]) loss)
    E.check (Map.singleton a (E.Node numeric (E.ObserveTaskLoss observed))) a === E.Unknown (E.UnsupportedObservation numeric)
    E.check (Map.singleton a (E.Node loss (E.Observe measured))) a === E.Unknown (E.UnsupportedObservation loss)

assumptions :: PropertyT IO ()
assumptions = do
    observed <- fixture
    let loss = E.TaskLoss (target observed)
        a = E.EvidenceId 0
        b = E.EvidenceId 1
        c = E.EvidenceId 2
    assumed <- accepted (E.check (Map.singleton a (E.Node loss E.Assume)) a)
    E.assumptions assumed === [loss]
    E.methods assumed === [E.Hypothesis]
    original <- accepted (U.finding (U.establish (target observed) observed))
    case E.assumptions original of
        premise : _ -> do
            let graph = Map.fromList [(a, E.Node loss (E.ObserveTaskLoss observed)), (b, E.Node premise E.Assume), (c, E.Node loss (E.Discharge a [b]))]
            stillConditional <- accepted (E.check graph c)
            assert (premise `elem` E.assumptions stillConditional)
        [] -> failure
