{-# LANGUAGE OverloadedStrings #-}

module Numerical (numerical) where

import Calls (change, field, request, wire)
import Control.Monad (forM_)
import Data.Aeson (Value (..), object, toJSON, (.=))
import Data.ByteString.Char8 qualified as Bytes
import Data.List (nub)
import Data.Map.Strict qualified as Map
import Data.Word (Word32)
import Hedgehog
import InferenceObservations (bound, fixture)
import Invar.Infer qualified as Infer
import Invar.Numerical qualified as N
import Invar.Spec.Evidence qualified as E
import Workloads (replace)

numerical :: Group
numerical =
    Group
        "Bound numerical observations"
        [ ("observations bind complete run identities and preserve exact measurements", once identities)
        , ("a divergence stops the common-prefix likelihood calculation", once divergence)
        , ("length and termination differences remain distinct observations", once termination)
        , ("raw signed zero survives observation even when the log ratio is zero", once signedZero)
        , ("invalid execution and prefix mismatches do not produce observations", once invalid)
        , ("individually valid runs must still share the declared paired inputs", once comparable)
        , ("finite findings retain individual external observation premises", once findings)
        , ("missing cross-scoring distributions and derivations are unknown", once insufficient)
        , ("graph rules cannot relabel scopes or discharge another scope", once scopes)
        , ("numerical counterexamples lift only through a declared conjunction", once witnesses)
        ]
  where
    once = withTests 1 . property

inputs :: PropertyT IO (N.Run, [Value])
inputs = do
    events <- fixture
    planned <- evalEither (Infer.prepare request)
    pure (N.Run planned bound 0 (wire events), events)

pair :: N.Run -> [Value] -> Either N.ObservationError N.Observed
pair before events = N.observe (N.BoundRun before before {N.logBytes = wire events})

alterResult :: (Value -> Value) -> [Value] -> [Value]
alterResult f = zipWith (\index event -> if index == (3 :: Int) then f event else event) [0 ..]

identities :: PropertyT IO ()
identities = do
    (run, events) <- inputs
    first <- evalEither (pair run events)
    N.tokensEqual (N.path first) === True
    N.behaviorBitsEqual (N.path first) === True
    N.firstDivergence (N.path first) === Nothing
    N.matchingSteps (N.path first) === 2
    N.prefixLogRatio (N.path first) === 0
    second <- evalEither (pair run (object ["stage" .= String "load"] : events))
    assert (N.scope first /= N.scope second)
    assert (N.scopeId (N.scope first) /= N.scopeId (N.scope second))
    N.path first === N.path second
    N.observe (N.BoundRun run run) === Right first

divergence :: PropertyT IO ()
divergence = do
    (run, events) <- inputs
    let different = alterResult (change "tokens" (toJSON [1, 2, 99 :: Int]) . probabilities [-0.25, -8] [0xbe800000, 0xc1000000]) events
    observed <- evalEither (pair run different)
    N.firstDivergence (N.path observed) === Just 1
    N.matchingSteps (N.path observed) === 1
    N.prefixLogRatio (N.path observed) === (-(1 / 4))
    immediate <- evalEither (pair run (alterResult (change "tokens" (toJSON [1, 99, 3 :: Int])) events))
    N.firstDivergence (N.path immediate) === Just 0
    N.prefixLogRatio (N.path immediate) === 0
    N.behaviorBitsEqual (N.path immediate) === False

termination :: PropertyT IO ()
termination = do
    (run, events) <- inputs
    short <- evalEither (pair run (alterResult (change "tokens" (toJSON [1, 2 :: Int]) . probabilities [-0.5] [0xbf000000] . change "truncated" (Bool False)) events))
    N.firstDivergence (N.path short) === Just 1
    N.referenceLength (N.path short) === 2
    N.candidateLength (N.path short) === 1
    stopped <- evalEither (pair run (alterResult (change "truncated" (Bool False)) events))
    N.firstDivergence (N.path stopped) === Nothing
    N.tokensEqual (N.path stopped) === True
    N.referenceTruncated (N.path stopped) === True
    N.candidateTruncated (N.path stopped) === False

signedZero :: PropertyT IO ()
signedZero = do
    (run, events) <- inputs
    let values word = alterResult (probabilities [0, -0.25] [word, 0xbe800000]) events
        positive = run {N.logBytes = wire (values 0)}
        negative = run {N.logBytes = replace "[0,-0.25]" "[-0.0,-0.25]" (wire (values 0x80000000))}
    observed <- evalEither (N.observe (N.BoundRun positive negative))
    N.behaviorBitsEqual (N.path observed) === False
    N.prefixLogRatio (N.path observed) === 0

invalid :: PropertyT IO ()
invalid = do
    (run, events) <- inputs
    N.observe (N.BoundRun run run {N.exitCode = 7}) === Left (N.ProcessFailed N.Candidate 7)
    forM_ [Bytes.empty, Bytes.init (wire events), wire (take 3 events)] $ \encoded ->
        case N.observe (N.BoundRun run run {N.logBytes = encoded}) of
            Left (N.InvalidRun N.Candidate _) -> success
            other -> annotateShow other >> failure
    pair run (alterResult (change "tokens" (toJSON [9, 2, 3 :: Int])) events) === Left (N.IncomparableInputs "tokenized prefix")

probabilities :: [Double] -> [Word32] -> Value -> Value
probabilities values bits = change "behavior" (toJSON values) . change "behavior_bits" (toJSON bits)

comparable :: PropertyT IO ()
comparable = do
    (run, events) <- inputs
    let variants =
            [ ("prompt", "prompt", request {Infer.prompt = "other"}, String "other")
            , ("temperature", "temperature", request {Infer.temperature = 1}, Number 1)
            , ("logical seed", "seed", request {Infer.seed = 18}, Number 18)
            , ("token budget", "tokens", request {Infer.tokens = 3}, Number 3)
            ]
    forM_ variants $ \(axis, key, requested, value) -> do
        planned <- evalEither (Infer.prepare requested)
        let changed = zipWith (\index event -> if index `elem` [1, 3 :: Int] then change "request" (change key value (field "request" event)) event else event) [0 ..] events
            candidate = run {N.planned = planned, N.logBytes = wire (alterResult (change "truncated" (Bool False)) changed)}
        N.observe (N.BoundRun run candidate) === Left (N.IncomparableInputs axis)

judge :: N.Relation -> N.Observed -> E.Verdict
judge relation observed = N.finding (N.establish (N.Claim (N.scope observed) relation) observed)

accepted :: E.Verdict -> PropertyT IO E.Certificate
accepted (E.Accept certificate) = pure certificate
accepted other = annotateShow other >> failure

findings :: PropertyT IO ()
findings = do
    (run, events) <- inputs
    observed <- evalEither (pair run events)
    forM_ [N.SameTokens, N.SameBehaviorBits, N.SameTermination, N.PrefixLogRatioWithin 0, N.PathLogRatioWithin 0] $ \relation -> do
        certificate <- accepted (judge relation observed)
        E.conclusion certificate === E.Numerical (N.Claim (N.scope observed) relation)
        length (E.assumptions certificate) === 6
        length (nub (E.assumptions certificate)) === 6
        E.methods certificate === [E.NumericalObservation]
    changed <- evalEither (pair run (alterResult (probabilities [-0.25, -0.25] [0xbe800000, 0xbe800000]) events))
    _ <- accepted (judge (N.PathLogRatioWithin (1 / 4)) changed)
    case judge (N.PathLogRatioWithin (1 / 8)) changed of
        E.Refute witness -> E.witness witness === E.NumericalWitness changed
        other -> annotateShow other >> failure

insufficient :: PropertyT IO ()
insufficient = do
    (run, events) <- inputs
    observed <- evalEither (pair run events)
    forM_ [N.ReferenceToCandidate, N.CandidateToReference] $ \direction ->
        judge (N.FullVocabularyKLWithin N.Reference direction 1) observed === E.Unknown (E.NumericalProblem N.MissingFullVocabulary)
    judge N.ModelSubstitution observed === E.Unknown (E.NumericalProblem N.UnsupportedModelDerivation)
    forM_ [N.PrefixLogRatioWithin (-1), N.PathLogRatioWithin (-1), N.FullVocabularyKLWithin N.Reference N.ReferenceToCandidate (-1)] $ \relation ->
        judge relation observed === E.Unknown (E.NumericalProblem (N.InvalidBudget (-1)))
    different <- evalEither (pair run (alterResult (change "tokens" (toJSON [1, 2, 9 :: Int])) events))
    judge (N.PathLogRatioWithin 100) different === E.Unknown (E.NumericalProblem N.MissingCrossScoring)
    stopped <- evalEither (pair run (alterResult (change "truncated" (Bool False)) events))
    judge (N.PathLogRatioWithin 100) stopped === E.Unknown (E.NumericalProblem N.MissingCrossScoring)

scopes :: PropertyT IO ()
scopes = do
    (run, events) <- inputs
    first <- evalEither (pair run events)
    second <- evalEither (pair run (object ["stage" .= String "load"] : events))
    let claim observed = E.Numerical (N.Claim (N.scope observed) N.SameTokens)
        a = E.EvidenceId 0
        b = E.EvidenceId 1
        c = E.EvidenceId 2
        different = E.NumericalProblem (N.ScopeMismatch (N.scopeId (N.scope first)) (N.scopeId (N.scope second)))
        graph = Map.fromList [(a, E.Node (claim first) (E.Observe first)), (b, E.Node (claim second) (E.Observe second))]
    N.finding (N.establish (N.Claim (N.scope first) N.SameTokens) second) === E.Unknown different
    E.check (Map.singleton a (E.Node (claim first) (E.Observe second))) a === E.Unknown different
    together <- accepted (E.check (Map.insert c (E.Node (E.All [claim first, claim second]) (E.Conjoin [a, b])) graph) c)
    length (E.assumptions together) === 12
    E.check (Map.insert c (E.Node (claim first) (E.Conjoin [a, b])) graph) c === E.Unknown (E.WrongConclusion (E.All [claim first, claim second]) (claim first))
    E.check (Map.insert c (E.Node (claim first) (E.Discharge a [b])) graph) c === E.Unknown (E.UnmatchedDischarge (claim second))
    let implication = E.Implies (claim first) (claim first)
        withImplication = Map.insert a (E.Node implication E.Assume) graph
    E.check (Map.insert c (E.Node (claim first) (E.Apply a b)) withImplication) c === E.Unknown (E.WrongConclusion (claim first) (claim second))

witnesses :: PropertyT IO ()
witnesses = do
    (run, events) <- inputs
    observed <- evalEither (pair run (alterResult (change "tokens" (toJSON [1, 2, 9 :: Int])) events))
    let target = E.Numerical (N.Claim (N.scope observed) N.SameTokens)
        first = E.EvidenceId 0
        second = E.EvidenceId 1
        together = E.All [target]
        graph = Map.fromList [(first, E.Node target (E.Observe observed)), (second, E.Node together (E.Conjoin [first]))]
    case E.check graph second of
        E.Refute result -> do
            E.refuted result === together
            E.witness result === E.NumericalWitness observed
        other -> annotateShow other >> failure
