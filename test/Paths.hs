{-# LANGUAGE OverloadedStrings #-}

module Paths (paths) where

import Data.Aeson (encode, object, (.=))
import Data.ByteString.Lazy qualified as Lazy
import Data.Either (isLeft)
import Data.Map.Strict qualified as Map
import Data.Maybe (listToMaybe)
import Data.Word (Word32)
import GHC.Float (castFloatToWord32)
import Hedgehog
import Invar.Learn.Report qualified as Report
import Invar.Paths qualified as P
import Invar.Spec.Evidence qualified as E
import Invar.Spec.Program qualified as Source
import Invar.Spec.Value qualified as Value
import Learning (learnerWorld, world)
import Probabilities (fixture)
import Updates (setup, setupFor, wire)

paths :: Group
paths =
    Group
        "Training and inference path observations"
        [ ("the same path and forward weights compare exact words", once compared)
        , ("a per-token bound catches a deviation that cancels in the path sum", once cancelled)
        , ("different paths, rules or forward weights are incomparable rather than violated", once incomparable)
        , ("a policy difference is reported without entering the comparison", once policyDifference)
        , ("a missing position and an invalid word leave the comparison undetermined", once missing)
        , ("zero support is an infinite difference only on one side", once infinite)
        , ("path and forward identities cover their declared contents", once identities)
        , ("an admitted update supplies observations and keeps cross-backend weights incomparable", once fromUpdate)
        , ("a learner-scored reference names the reference adapter on both sides", once reference)
        , ("a path claim is bound to one observation scope", once scoped)
        , ("a path observation enters the evidence graph with its external premises", once evidence)
        , ("current observations use the adapter identity of their own optimizer step", once currentSteps)
        , ("a current step that changes state but not weights remains comparable", once currentComparable)
        ]
  where
    once = withTests 1 . property

word :: Float -> Word32
word = castFloatToWord32

weights :: String -> String -> P.ForwardWeights
weights = P.ForwardWeights

baseA, baseB, adapter :: String
baseA = replicate 64 '1'
baseB = replicate 64 '2'
adapter = replicate 64 '3'

forwardA, forwardB :: P.ForwardWeights
forwardA = weights baseA adapter
forwardB = weights baseB adapter

input :: P.Input
input = P.Input "Compute the answer." 17 1 0.8

path :: P.TokenPath
path = P.TokenPath [10, 11] [20, 21, 22]

rule :: P.Rule
rule = P.Rule 0.8 "processed" "none"

observation :: P.ForwardWeights -> [Word32] -> P.Observation
observation forward words32 = P.Observation (P.pathId path) rule input forward (P.ImplementationId "learner") Nothing words32 (P.Learner 0 0 "s0" P.Proximal)

claim :: P.Relation -> P.Claim
claim = P.Claim

compared :: PropertyT IO ()
compared = do
    let left = observation forwardA [word (-1), word (-2), word (-0.5)]
        right = observation forwardA [word (-1), word (-1.5), word (-0.75)]
    gap <- evalEither (P.measure left right)
    P.tokens gap === 3
    P.identical gap === 1
    P.largest gap === P.Finite (1 / 2)
    P.ratio gap === P.Finite (-(1 / 4))
    P.judge (claim (P.PerTokenLogGapWithin (1 / 2))) left right === P.Satisfied gap
    P.judge (claim (P.PerTokenLogGapWithin (1 / 4))) left right === P.Violated gap
    P.judge (claim (P.PathLogRatioWithin (1 / 4))) left right === P.Satisfied gap
    sameGap <- evalEither (P.measure left left)
    P.judge (claim P.SameWords) left left === P.Satisfied sameGap

cancelled :: PropertyT IO ()
cancelled = do
    let left = observation forwardA [word (-1), word (-3)]
        right = observation forwardA [word (-2), word (-2)]
    gap <- evalEither (P.measure left right)
    P.ratio gap === P.Finite 0
    P.largest gap === P.Finite 1
    P.judge (claim (P.PathLogRatioWithin 0)) left right === P.Satisfied gap
    P.judge (claim (P.PerTokenLogGapWithin (1 / 2))) left right === P.Violated gap

incomparable :: PropertyT IO ()
incomparable = do
    let left = observation forwardA [word (-1)]
        otherPath = left {P.path = P.pathId (P.TokenPath [10, 11] [20, 22])}
        otherRule = left {P.rule = P.Rule 0.7 "processed" "none"}
        otherWeights = observation forwardB [word (-1)]
        otherAssembly = left {P.implementation = P.ImplementationId "another-assembly"}
    P.judge (claim P.SameWords) left otherPath === P.Insufficient (P.PathDiffer (P.path left) (P.path otherPath))
    P.judge (claim P.SameWords) left otherRule === P.Insufficient (P.RuleDiffer rule (P.Rule 0.7 "processed" "none"))
    P.judge (claim P.SameWords) left otherWeights === P.Insufficient (P.ForwardDiffer forwardA forwardB)
    gap <- evalEither (P.measure left otherAssembly)
    P.judge (claim P.SameWords) left otherAssembly === P.Satisfied gap

policyDifference :: PropertyT IO ()
policyDifference = do
    let left = observation forwardA [word (-1)]
        right = observation forwardB [word (-1)]
    assert (P.policyDifference P.SameWords left right)
    assert (not (P.policyDifference P.SameWords left left))
    assert (not (P.policyDifference P.SameWords left (left {P.path = P.pathId (P.TokenPath [10] [20])})))

missing :: PropertyT IO ()
missing = do
    let left = observation forwardA [word (-1), word (-2)]
        shorter = observation forwardA [word (-1)]
        invalid = observation forwardA [0]
    assert (isLeft (P.measure left shorter))
    assert (isLeft (P.measure left invalid))

infinite :: PropertyT IO ()
infinite = do
    let zeroSupport = 0xff800000
        left = observation forwardA [zeroSupport]
        finite = observation forwardA [word (-1)]
        both = observation forwardA [zeroSupport]
    gap <- evalEither (P.measure left finite)
    P.largest gap === P.PositiveInfinity
    P.ratio gap === P.PositiveInfinity
    P.judge (claim (P.PerTokenLogGapWithin 1)) left finite === P.Violated gap
    same <- evalEither (P.measure left both)
    P.judge (claim P.SameWords) left both === P.Satisfied same

identities :: PropertyT IO ()
identities = do
    P.pathId path /== P.pathId (P.TokenPath [10, 12] [20, 21, 22])
    P.pathId path /== P.pathId (P.TokenPath [10] [11, 20, 21, 22])
    P.forwardId forwardA /== P.forwardId forwardB
    P.forwardId forwardA /== P.forwardId (weights baseA (replicate 64 '4'))

learnerSide, engineSide :: P.Side
learnerSide = P.Side "test-model" "test-revision" (replicate 64 'd') (replicate 64 'e') (replicate 64 'f')
engineSide = P.Side "test-model" "test-revision" (replicate 64 'd') (replicate 64 '0') (replicate 64 '1')

startPolicy :: String
startPolicy = replicate 64 'a'

fromUpdate :: PropertyT IO ()
fromUpdate = do
    (_, events) <- setup
    report <- evalEither (Report.admit 7 (wire events))
    observed <- evalEither (P.fromUpdate learnerSide engineSide report (Lazy.toStrict (encode (fixture events))) [(0, "s0", startPolicy), (0, "s1", startPolicy)])
    let pairs = P.proximal observed
    assert (not (null pairs))
    gap <- evalEither (P.measureAll pairs)
    P.tokens gap === 2
    P.identical gap === 2
    P.ratio gap === P.Finite 0
    P.describeGap gap === object ["tokens" .= (2 :: Int), "identical" .= (2 :: Int), "gap" .= object ["p50" .= (0 :: Double), "p90" .= (0 :: Double), "p99" .= (0 :: Double), "mean" .= (0 :: Double), "mean_absolute" .= (0 :: Double), "max" .= (0 :: Double)], "path_log_ratio" .= object ["kind" .= ("finite" :: String), "numerator" .= (0 :: Int), "denominator" .= (1 :: Int)]]
    (learner, engine) <- evalMaybe (listToMaybe pairs)
    assert (P.policyDifference P.SameWords learner engine)
    P.judge (claim P.SameWords) learner engine === P.Insufficient (P.ForwardDiffer (P.forward learner) (P.forward engine))
    P.forward learner === P.ForwardWeights (replicate 64 'e') startPolicy
    P.forward engine === P.ForwardWeights (replicate 64 '0') startPolicy
    P.input learner === P.Input "Compute the answer." 18 1 0.8

reference :: PropertyT IO ()
reference = do
    (_, events) <- setupFor learnerWorld
    report <- evalEither (Report.admit 7 (wire events))
    observed <- evalEither (P.fromUpdate learnerSide engineSide report (Lazy.toStrict (encode (fixture events))) [(0, "s0", startPolicy), (0, "s1", startPolicy)])
    (learner, engine) <- evalMaybe (listToMaybe (P.references observed))
    P.forward learner === P.ForwardWeights (replicate 64 'e') (replicate 64 'c')
    P.forward engine === P.ForwardWeights (replicate 64 '0') (replicate 64 'c')
    P.provenanceName learner === "s0"
    P.provenanceName engine === "s0"

scoped :: PropertyT IO ()
scoped = do
    let left = observation forwardA [word (-1)]
        right = observation forwardA [word (-1)]
        observed = P.scoped left right
    gap <- evalEither (P.measure left right)
    P.judgeScoped (P.ScopedClaim (P.scope observed) P.SameWords) observed === P.Satisfied gap
    P.judgeScoped (P.ScopedClaim (P.ScopeId "another-scope") P.SameWords) observed === P.Insufficient (P.ScopeMismatch (P.ScopeId "another-scope") (P.scope observed))

evidence :: PropertyT IO ()
evidence = do
    let left = observation forwardA [word (-1)]
        right = observation forwardA [word (-1)]
        observed = P.scoped left right
        target = E.Paths (P.ScopedClaim (P.scope observed) P.SameWords)
        graph = Map.singleton (E.EvidenceId 0) (E.Node target (E.ObservePaths observed))
    case E.check graph (E.EvidenceId 0) of
        E.Accept certificate -> do
            E.conclusion certificate === target
            E.methods certificate === [E.PathObservation]
            length (E.assumptions certificate) === 4
        other -> annotateShow other >> failure

currentSteps :: PropertyT IO ()
currentSteps = do
    let algorithm = Value.Record (Map.fromList [("epsilon", Value.Atom (Value.Number (1 / 5))), ("penalty", Value.Atom (Value.Number (1 / 25))), ("delta", Value.Atom (Value.Number (1 / 10000))), ("steps", Value.Atom (Value.Number 2))])
        sameReference = Map.insert (Source.Semantic "reference_scores") (world Map.! Source.Semantic "behavior") world
    (_, events) <- setupFor (Map.insert (Source.Semantic "algorithm") algorithm sameReference)
    report <- evalEither (Report.admit 7 (wire events))
    observed <- evalEither (P.fromUpdate learnerSide engineSide report (Lazy.toStrict (encode (fixture events))) [(0, "s0", startPolicy), (0, "s1", "1" <> replicate 63 '1'), (1, "s0", startPolicy), (1, "s1", "1" <> replicate 63 '1')])
    length (P.currents observed) === 2
    (firstLearner, firstEngine) <- evalMaybe (listToMaybe (P.currents observed))
    (secondLearner, secondEngine) <- evalMaybe (listToMaybe (drop 1 (P.currents observed)))
    assert (P.provenanceName firstLearner == P.provenanceName firstEngine)
    assert (P.provenanceName secondLearner == P.provenanceName secondEngine)
    P.judge (claim P.SameWords) firstLearner firstEngine === P.Insufficient (P.ForwardDiffer (P.forward firstLearner) (P.forward firstEngine))
    P.judge (claim P.SameWords) secondLearner secondEngine === P.Insufficient (P.ForwardDiffer (P.forward secondLearner) (P.forward secondEngine))
    P.forward secondLearner === P.ForwardWeights (replicate 64 'e') ("1" <> replicate 63 '1')

currentComparable :: PropertyT IO ()
currentComparable = do
    let learner = (observation forwardA [word (-1)]) {P.provenance = P.Learner 0 1 "s0" P.Current}
        engine = (observation forwardA [word (-1)]) {P.provenance = P.Rollout 0 "s0"}
    gap <- evalEither (P.measure learner engine)
    P.judge (claim P.SameWords) learner engine === P.Satisfied gap
    assert (not (P.policyDifference P.SameWords learner engine))
