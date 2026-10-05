{-# LANGUAGE OverloadedStrings #-}

module NumericalScores (numericalScores) where

import Calls (change, field, request, wire)
import Control.Monad (forM_)
import Data.Aeson (Value (..), object, toJSON, (.=))
import Data.List (nub)
import Data.Map.Strict qualified as Map
import Data.Word (Word32)
import Hedgehog
import InferenceObservations qualified as Source
import Invar.Infer qualified as Infer
import Invar.Numerical qualified as N
import Invar.Score qualified as Score
import Invar.Spec.Evidence qualified as E
import Scores qualified

numericalScores :: Group
numericalScores =
    Group
        "Scored numerical evidence"
        [ ("diverged paths have separate finite directional judgments", once directions)
        , ("missing directions and non-path claims cannot be inferred", once missing)
        , ("zero target support refutes every finite selected-path budget", once zeroSupport)
        , ("individually admitted scores still require matching pair sources and targets", once correspondence)
        , ("duplicate attachments fail and canonical order preserves scope", once attachments)
        , ("scored scope and individual premises survive evidence rules", once scopeRules)
        ]
  where
    once = withTests 1 . property

fixture :: PropertyT IO (N.Run, N.Run, [Value], [Value])
fixture = do
    original <- Source.fixture
    planned <- evalEither (Infer.prepare request)
    let other = alterResult (change "tokens" (toJSON [1, 9, 3 :: Int])) original
        run events = N.Run planned Source.bound 0 (wire events) Nothing
    pure (run original, run other, original, other)

alterResult :: (Value -> Value) -> [Value] -> [Value]
alterResult modify = map (\value -> if field "stage" value == String "result" then modify value else value)

measured :: [Value] -> [Word32] -> PropertyT IO Score.Report
measured events bits = do
    (call, scoreEvents) <- Scores.fixtureFor events request
    evalEither (Score.admit call 0 (wire (Scores.changeObservation (change "log_probability_bits" (toJSON bits)) scoreEvents)))

judge :: N.Relation -> N.Observed -> E.Verdict
judge relation observed = N.finding (N.establish (N.Claim (N.scope observed) relation) observed)

accepted :: E.Verdict -> PropertyT IO E.Certificate
accepted (E.Accept certificate) = pure certificate
accepted other = annotateShow other >> failure

refuted :: N.Observed -> E.Verdict -> PropertyT IO ()
refuted expected (E.Refute result) = E.witness result === E.NumericalWitness expected
refuted _ other = annotateShow other >> failure

directions :: PropertyT IO ()
directions = do
    (left, right, original, other) <- fixture
    forward <- measured original [0xbf800000, 0xc0000000]
    backward <- measured other [0, 0xbe800000]
    Score.logRatio forward === Score.Finite (9 / 4)
    Score.logRatio backward === Score.Finite (-(1 / 2))
    observed <- evalEither (N.observe (N.ScoredRun left right [(N.Reference, forward), (N.Candidate, backward)]))
    N.firstDivergence (N.path observed) === Just 0
    N.prefixLogRatio (N.path observed) === 0
    _ <- accepted (judge (N.ScoredPathLogRatioWithin N.Reference (9 / 4)) observed)
    _ <- accepted (judge (N.ScoredPathLogRatioWithin N.Candidate (1 / 2)) observed)
    refuted observed (judge (N.ScoredPathLogRatioWithin N.Reference 2) observed)
    refuted observed (judge (N.ScoredPathLogRatioWithin N.Candidate (1 / 4)) observed)

missing :: PropertyT IO ()
missing = do
    (left, right, original, _) <- fixture
    forward <- measured original [0xbf800000, 0xc0000000]
    bare <- evalEither (N.observe (N.BoundRun left right))
    observed <- evalEither (N.observe (N.ScoredRun left right [(N.Reference, forward)]))
    forM_ [N.Reference, N.Candidate] $ \side -> do
        judge (N.ScoredPathLogRatioWithin side 10) bare === E.Unknown (E.NumericalProblem (N.MissingScoredPath side))
        judge (N.ScoredPathLogRatioWithin side (-1)) observed === E.Unknown (E.NumericalProblem (N.InvalidBudget (-1)))
    judge (N.ScoredPathLogRatioWithin N.Candidate 10) observed === E.Unknown (E.NumericalProblem (N.MissingScoredPath N.Candidate))
    judge (N.PathLogRatioWithin 10) observed === E.Unknown (E.NumericalProblem N.MissingCrossScoring)
    forM_ [N.ReferenceToCandidate, N.CandidateToReference] $ \direction ->
        judge (N.FullVocabularyKLWithin N.Reference direction 10) observed === E.Unknown (E.NumericalProblem N.MissingFullVocabulary)

zeroSupport :: PropertyT IO ()
zeroSupport = do
    (left, right, original, other) <- fixture
    forward <- measured original [0xff800000, 0xbe800000]
    backward <- measured other [0xbf000000, 0xff800000]
    forM_ [(N.Reference, forward), (N.Candidate, backward)] $ \(side, score) -> do
        Score.logRatio score === Score.PositiveInfinity
        observed <- evalEither (N.observe (N.ScoredRun left right [(side, score)]))
        forM_ [0, 1, 10 ^ (20 :: Int)] $ \budget -> refuted observed (judge (N.ScoredPathLogRatioWithin side budget) observed)

correspondence :: PropertyT IO ()
correspondence = do
    (left, right, original, _) <- fixture
    forward <- measured original [0xbf800000, 0xc0000000]
    N.observe (N.ScoredRun left right [(N.Candidate, forward)]) === Left (N.InvalidScore N.Candidate "source execution")
    let changedSource = left {N.logBytes = wire (map (\event -> if field "stage" event == String "load" then change "cpu_seconds" (Number 3) event else event) original)}
    N.observe (N.ScoredRun changedSource right [(N.Reference, forward)]) === Left (N.InvalidScore N.Reference "source execution")
    (wrongCall, wrongEvents) <- Scores.fixtureFor original request {Infer.base = replicate 64 '0'}
    wrongTarget <- evalEither (Score.admit wrongCall 0 (wire wrongEvents))
    N.observe (N.ScoredRun left right [(N.Reference, wrongTarget)]) === Left (N.InvalidScore N.Reference "target request")
    (call, events) <- Scores.fixtureFor original request
    let wrongLoaded = map (\value -> if field "stage" value == String "loaded_adapter" then change "revision" (String "other") value else value) events
        wrongDescription = Scores.changeObservation (\value -> change "target" (change "revision" (String "other") (field "target" value)) value) wrongLoaded
    wrongPolicy <- evalEither (Score.admit call 0 (wire wrongDescription))
    N.observe (N.ScoredRun left right [(N.Reference, wrongPolicy)]) === Left (N.InvalidScore N.Reference "target materialization")

attachments :: PropertyT IO ()
attachments = do
    (left, right, original, other) <- fixture
    forward <- measured original [0xbf800000, 0xc0000000]
    backward <- measured other [0, 0xbe800000]
    let supplied = [(N.Reference, forward), (N.Candidate, backward)]
    first <- evalEither (N.observe (N.ScoredRun left right supplied))
    N.observe (N.ScoredRun left right (reverse supplied)) === Right first
    N.observe (N.ScoredRun left right [(N.Reference, forward), (N.Reference, forward)]) === Left (N.InvalidScore N.Reference "duplicate path scores")

scopeRules :: PropertyT IO ()
scopeRules = do
    (left, right, original, _) <- fixture
    (call, events) <- Scores.fixtureFor original request
    forward <- evalEither (Score.admit call 0 (wire events))
    another <- evalEither (Score.admit call 0 (wire (object ["stage" .= String "load"] : events)))
    first <- evalEither (N.observe (N.ScoredRun left right [(N.Reference, forward)]))
    second <- evalEither (N.observe (N.ScoredRun left right [(N.Reference, another)]))
    assert (N.scope first /= N.scope second)
    assert (N.scopeId (N.scope first) /= N.scopeId (N.scope second))
    let relation = N.ScoredPathLogRatioWithin N.Reference 0
        claim value = E.Numerical (N.Claim (N.scope value) relation)
        a = E.EvidenceId 0
        b = E.EvidenceId 1
        c = E.EvidenceId 2
        graph = Map.fromList [(a, E.Node (claim first) (E.Observe first)), (b, E.Node (claim second) (E.Observe second))]
    certificate <- accepted (judge relation first)
    length (E.assumptions certificate) === 10
    length (nub (E.assumptions certificate)) === 10
    let obligations = [value | E.External value <- E.assumptions certificate]
    length (filter ((== "selected-token-behavior/v1") . E.observation) obligations) === 6
    length (filter ((== "selected-path-behavior/v1") . E.observation) obligations) === 4
    length (filter ((== "cached-path-score/v1") . E.specification) obligations) === 4
    E.methods certificate === [E.NumericalObservation]
    N.finding (N.establish (N.Claim (N.scope first) relation) second) === E.Unknown (E.NumericalProblem (N.ScopeMismatch (N.scopeId (N.scope first)) (N.scopeId (N.scope second))))
    together <- accepted (E.check (Map.insert c (E.Node (E.All [claim first, claim second]) (E.Conjoin [a, b])) graph) c)
    length (E.assumptions together) === 20
