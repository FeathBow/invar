{-# LANGUAGE OverloadedStrings #-}

module NumericalProbes (numericalProbes) where

import Calls (change, field, request, wire)
import Control.Monad (forM_)
import Data.Aeson (Value (..), object, toJSON, (.=))
import Data.Aeson.Types (parseEither, parseJSON)
import Data.Foldable (toList)
import Data.List (nub)
import Data.Map.Strict qualified as Map
import Data.Ratio ((%))
import GHC.Float (castFloatToWord32)
import Hedgehog
import InferenceObservations qualified as Source
import Invar.Infer qualified as Infer
import Invar.Numerical qualified as N
import Invar.Score qualified as Score
import Invar.Spec.Evidence qualified as E
import Numeric.Natural (Natural)
import ScoreProbes qualified
import Scores qualified

numericalProbes :: Group
numericalProbes =
    Group
        "Full-vocabulary numerical evidence"
        [ ("KL budgets use an outward interval and retain boundary uncertainty", once boundaries)
        , ("source path and probability direction remain distinct", once directions)
        , ("zero support outside the selected token can produce infinite KL", once support)
        , ("both target probes and identical inventories are required", once correspondence)
        , ("the full domain must pass and a violation outranks uncertainty", once allSteps)
        , ("scope retains raw vectors source identity and every external premise", once scopeRules)
        ]
  where
    once = withTests 1 . property

fixture :: PropertyT IO (N.Run, N.Run, [Value], [Value])
fixture = do
    original <- Source.fixture
    planned <- evalEither (Infer.prepare request)
    let other = map (\value -> if field "stage" value == String "result" then change "tokens" (toJSON [1, 3, 2 :: Int]) value else value) original
        run events = N.Run planned Source.bound 0 (wire events)
    pure (run original, run other, original, other)

measured :: [Value] -> [(Natural, [Float])] -> PropertyT IO Score.Report
measured original snapshots = do
    (call, events) <- ScoreProbes.fixtureFor (map fst snapshots) original request
    let retained = [object ["step" .= step, "probability_bits" .= map castFloatToWord32 values] | (step, values) <- snapshots]
        modify value = change "full_vocabulary" (change "snapshots" (toJSON retained) (field "full_vocabulary" value)) value
    evalEither (Score.admit call 0 (wire (Scores.changeObservation modify events)))

pair :: [Value] -> [Float] -> PropertyT IO [N.Probe]
pair original right = do
    before <- measured original [(0, replicate 4 0.25)]
    after <- measured original [(0, right)]
    pure [N.Probe N.Reference N.Reference before, N.Probe N.Reference N.Candidate after]

judge :: (N.Side, N.Direction) -> Rational -> N.Observed -> E.Verdict
judge (side, direction) budget observed = N.finding (N.establish (N.Claim (N.scope observed) (N.FullVocabularyKLWithin side direction budget)) observed)

accepted :: E.Verdict -> PropertyT IO E.Certificate
accepted (E.Accept value) = pure value
accepted other = annotateShow other >> failure

refuted :: N.Observed -> E.Verdict -> PropertyT IO ()
refuted expected (E.Refute result) = E.witness result === E.NumericalWitness expected
refuted _ other = annotateShow other >> failure

interval :: N.Observed -> PropertyT IO (Rational, Rational)
interval observed = do
    selectedPath <- firstValue (field "full_vocabulary" (N.describe observed))
    selectedStep <- firstValue (field "steps" selectedPath)
    let bounds = field "kl_reference_candidate" selectedStep
    (,) <$> rational (field "lower" bounds) <*> rational (field "upper" bounds)
  where
    firstValue (Array values) | selected : _ <- toList values = pure selected
    firstValue other = annotateShow other >> failure
    rational value = do
        numerator <- evalEither (parseEither parseJSON (field "numerator" value))
        denominator <- evalEither (parseEither parseJSON (field "denominator" value))
        pure (numerator % denominator)

boundaries :: PropertyT IO ()
boundaries = do
    (left, right, original, _) <- fixture
    probes <- pair original [0.125, 0.125, 0.25, 0.5]
    observed <- evalEither (N.observe (N.ProbedRun left right [] probes))
    (lower, upper) <- interval observed
    assert (0 < lower && lower < upper)
    _ <- accepted (judge (N.Reference, N.ReferenceToCandidate) upper observed)
    refuted observed (judge (N.Reference, N.ReferenceToCandidate) (lower / 2) observed)
    judge (N.Reference, N.ReferenceToCandidate) ((lower + upper) / 2) observed === E.Unknown (E.NumericalProblem (N.KLReductionUncertain N.Reference N.ReferenceToCandidate [0]))
    judge (N.Reference, N.ReferenceToCandidate) (-1) observed === E.Unknown (E.NumericalProblem (N.InvalidBudget (-1)))
    proportional <- pair original (replicate 4 0.125)
    equal <- evalEither (N.observe (N.ProbedRun left right [] proportional))
    interval equal >>= (=== (0, 0))
    _ <- accepted (judge (N.Reference, N.ReferenceToCandidate) 0 equal)
    pure ()

directions :: PropertyT IO ()
directions = do
    (left, right, original, other) <- fixture
    probes <- pair original [0.125, 0.125, 0.25, 0.5]
    observed <- evalEither (N.observe (N.ProbedRun left right [] probes))
    -- The second pair has distinct forward and reverse KL values.
    unequal <- pair original [0.0625, 0.0625, 0.0625, 0.8125]
    asymmetric <- evalEither (N.observe (N.ProbedRun left right [] unequal))
    _ <- accepted (judge (N.Reference, N.CandidateToReference) (7 / 10) asymmetric)
    refuted asymmetric (judge (N.Reference, N.ReferenceToCandidate) (7 / 10) asymmetric)
    judge (N.Candidate, N.ReferenceToCandidate) 1 observed === E.Unknown (E.NumericalProblem N.MissingFullVocabulary)
    candidateBefore <- measured other [(0, replicate 4 0.25)]
    candidateAfter <- measured other [(0, replicate 4 0.25)]
    both <- evalEither (N.observe (N.ProbedRun left right [] (probes ++ [N.Probe N.Candidate N.Reference candidateBefore, N.Probe N.Candidate N.Candidate candidateAfter])))
    _ <- accepted (judge (N.Candidate, N.ReferenceToCandidate) 0 both)
    refuted both (judge (N.Reference, N.ReferenceToCandidate) 0 both)

support :: PropertyT IO ()
support = do
    (left, right, original, _) <- fixture
    probes <- pair original [0, 0.25, 0.25, 0.5]
    observed <- evalEither (N.observe (N.ProbedRun left right [] probes))
    forM_ [0, 1, 10 ^ (30 :: Int)] $ \budget -> refuted observed (judge (N.Reference, N.ReferenceToCandidate) budget observed)
    _ <- accepted (judge (N.Reference, N.CandidateToReference) 1 observed)
    pure ()

correspondence :: PropertyT IO ()
correspondence = do
    (left, right, original, _) <- fixture
    probes <- pair original [0.125, 0.125, 0.25, 0.5]
    partial <- evalEither (N.observe (N.ProbedRun left right [] (take 1 probes)))
    judge (N.Reference, N.ReferenceToCandidate) 1 partial === E.Unknown (E.NumericalProblem N.MissingFullVocabulary)
    N.observe (N.ProbedRun left right [] (probes ++ take 1 probes)) === Left (N.InvalidProbe N.Reference "duplicate probe target")
    N.observe (N.ProbedRun left right [] [probe {N.pathSource = N.Candidate} | probe <- probes]) === Left (N.InvalidProbe N.Candidate "source execution")
    later <- measured original [(1, replicate 4 0.25)]
    N.observe (N.ProbedRun left right [] (take 1 probes ++ [N.Probe N.Reference N.Candidate later])) === Left (N.InvalidProbe N.Reference "different probe steps")
    (wideCall, wideEvents) <- ScoreProbes.fixtureFor [0] original request
    let wider value =
            change
                "full_vocabulary"
                ( foldr
                    (uncurry change)
                    (field "full_vocabulary" value)
                    [("vocabulary", Number 5), ("raw_payload_bytes", Number 20), ("snapshots", toJSON [object ["step" .= (0 :: Int), "probability_bits" .= map castFloatToWord32 [0.25, 0.25, 0.25, 0.25, 0]]])]
                )
                value
    wide <- evalEither (Score.admit wideCall 0 (wire (Scores.changeObservation wider wideEvents)))
    N.observe (N.ProbedRun left right [] (take 1 probes ++ [N.Probe N.Reference N.Candidate wide])) === Left (N.InvalidProbe N.Reference "different vocabulary widths")
    (plainCall, plainEvents) <- Scores.fixtureFor original request
    plain <- evalEither (Score.admit plainCall 0 (wire plainEvents))
    N.observe (N.ProbedRun left right [] [N.Probe N.Reference N.Reference plain]) === Left (N.InvalidProbe N.Reference "missing full-vocabulary snapshots")
    (call, events) <- ScoreProbes.fixtureFor [0] original request {Infer.base = replicate 64 '0'}
    wrongTarget <- evalEither (Score.admit call 0 (wire events))
    N.observe (N.ProbedRun left right [] [N.Probe N.Reference N.Reference wrongTarget]) === Left (N.InvalidProbe N.Reference "target request")

allSteps :: PropertyT IO ()
allSteps = do
    (left, right, original, _) <- fixture
    before <- measured original [(0, replicate 4 0.25), (1, replicate 4 0.25)]
    after <- measured original [(0, [0.125, 0.125, 0.25, 0.5]), (1, [0.03125, 0.03125, 0.03125, 0.90625])]
    observed <- evalEither (N.observe (N.ProbedRun left right [] [N.Probe N.Reference N.Reference before, N.Probe N.Reference N.Candidate after]))
    (lower, upper) <- interval observed
    refuted observed (judge (N.Reference, N.ReferenceToCandidate) ((lower + upper) / 2) observed)
    _ <- accepted (judge (N.Reference, N.ReferenceToCandidate) 10 observed)
    pure ()

scopeRules :: PropertyT IO ()
scopeRules = do
    (left, right, original, _) <- fixture
    probes <- pair original [0.125, 0.125, 0.25, 0.5]
    observed <- evalEither (N.observe (N.ProbedRun left right [] probes))
    N.observe (N.ProbedRun left right [] (reverse probes)) === Right observed
    changed <- pair original [0.0625, 0.0625, 0.125, 0.25]
    another <- evalEither (N.observe (N.ProbedRun left right [] changed))
    assert (N.scope observed /= N.scope another)
    assert (N.scopeId (N.scope observed) /= N.scopeId (N.scope another))
    certificate <- accepted (judge (N.Reference, N.ReferenceToCandidate) 1 observed)
    length (E.assumptions certificate) === 16
    length (nub (E.assumptions certificate)) === 16
    let obligations = [value | E.External value <- E.assumptions certificate]
    length (filter ((== "selected-token-behavior/v1") . E.observation) obligations) === 6
    length (filter ((== "full-vocabulary-behavior/v1") . E.observation) obligations) === 10
    length (filter ((== "cached-distribution-probe/v1") . E.specification) obligations) === 10
    let relation = N.FullVocabularyKLWithin N.Reference N.ReferenceToCandidate 1
        claim value = E.Numerical (N.Claim (N.scope value) relation)
        a = E.EvidenceId 0
        b = E.EvidenceId 1
        c = E.EvidenceId 2
        graph = Map.fromList [(a, E.Node (claim observed) (E.Observe observed)), (b, E.Node (claim another) (E.Observe another))]
    N.finding (N.establish (N.Claim (N.scope observed) relation) another) === E.Unknown (E.NumericalProblem (N.ScopeMismatch (N.scopeId (N.scope observed)) (N.scopeId (N.scope another))))
    E.check (Map.insert c (E.Node (claim observed) (E.Discharge a [b])) graph) c === E.Unknown (E.UnmatchedDischarge (claim another))
