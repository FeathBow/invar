{-# LANGUAGE OverloadedStrings #-}

module Steps (steps) where

import Control.Monad (forM_, zipWithM)
import Data.List (zipWith5)
import Data.Map.Strict qualified as Map
import Data.Word (Word32)
import GHC.Float (castFloatToWord32)
import Hedgehog
import Invar.Async.Completion qualified as Completion
import Invar.Learn.Algorithm qualified as Definition
import Invar.Learn.Objective qualified as O
import Invar.Learn.Stream (Applied (..), Current (..), Reply (..), Sample (..))
import Invar.Learn.Stream qualified as S
import Invar.Spec.Invocation qualified as V

steps :: Group
steps =
    Group
        "Per-step learner exchange"
        [ ("one full step reuses each first observation as proximal and answers with the core's cotangents", once full)
        , ("later steps keep proximal fixed and chain the learner state", once staged)
        , ("samples outside the first step need proximal before the first update", once outside)
        , ("learner-scored reference words arrive before the step that consumes them and drive the objective", once learnerReference)
        , ("reports out of order, from another state or with other cotangents are refused", once refusals)
        , ("the cotangent digest is SHA-256 over little-endian FP32 words", once vector)
        , ("a stream starts only from a valid plan, and its identity names the plan", once plans)
        , ("a distilled stream answers each token with the teacher word over the frozen proximal word and refuses an invalid teacher word", once distilled)
        ]
  where
    once = withTests 1 . property

word :: Float -> Word32
word = castFloatToWord32

exchange :: V.Binding
exchange = V.Binding (V.CallId 1) (V.AttemptId 2) (V.Instance 3)

profile :: O.Profile
profile = O.Profile 0.2 0.04

first', second' :: Sample
first' = Sample "a" [word (-1), word (-0.5)] [] (S.Advantage (word 1))
second' = Sample "b" [word (-2)] [word (-1.5)] (S.Advantage (word (-1)))

planned :: S.Plan
planned = S.Plan profile "p0" [first', second'] [["a", "b"]] S.FromEngine S.Grouped

taught :: S.Plan
taught = planned {S.members = [first' {S.signal = S.Teacher [word (-0.4), word (-2)]}, second' {S.signal = S.Teacher [word (-1)]}], S.batches = [["a"], ["b"]], S.algorithm = S.Distilled teacher}

teacher :: Definition.Teacher
teacher = Definition.Teacher "teacher-model" "teacher-revision" (replicate 64 'd') (replicate 64 '7') (replicate 64 '8') (replicate 64 '9') 1 "none" "processed" "forced"

expected :: [Word32] -> [Word32] -> Sample -> Int -> PropertyT IO [O.Output]
expected now old entry = scoredWith (if null (referenceWords entry) then behaviorWords entry else referenceWords entry) now old entry

advantagesOf :: Sample -> [Word32] -> PropertyT IO [Word32]
advantagesOf entry old = case signal entry of
    S.Advantage advantage -> pure (map (const advantage) old)
    S.Teacher words32 -> evalEither (zipWithM O.distilled words32 old)

full :: PropertyT IO ()
full = do
    begun <- evalEither (S.begin exchange planned)
    let nowA = [word (-0.9), word (-0.6)]
        nowB = [word (-1.8)]
    (afterA, replyA) <- evalEither (S.current begun (Current 0 "a" nowA (S.observationOf nowA) "p0"))
    outputsA <- expected nowA nowA first' 3
    objective replyA === map O.gradient outputsA
    surrogate replyA === map O.surrogateGradient outputsA
    (afterB, replyB) <- evalEither (S.current afterA (Current 0 "b" nowB (S.observationOf nowB) "p0"))
    outputsB <- expected nowB nowB second' 3
    objective replyB === map O.gradient outputsB
    S.proximals afterB === Map.fromList [("a", nowA), ("b", nowB)]
    (done, closed) <- evalEither (S.applied afterB (Applied 0 "p0" "p1" [S.digest replyA, S.digest replyB]))
    (Completion.step closed, Completion.before closed, Completion.after closed) === (0, "p0", "p1")
    S.complete done === Right "p1"
    S.current done (Current 1 "a" nowA (S.observationOf nowA) "p1") === Left S.Finished

staged :: PropertyT IO ()
staged = do
    begun <- evalEither (S.begin exchange (planned {S.batches = [["a", "b"], ["a"]]}))
    let firstA = [word (-0.9), word (-0.6)]
        laterA = [word (-0.7), word (-0.8)]
    (one, replyA) <- evalEither (S.current begun (Current 0 "a" firstA (S.observationOf firstA) "p0"))
    (two, replyB) <- evalEither (S.current one (Current 0 "b" [word (-1.8)] (S.observationOf [word (-1.8)]) "p0"))
    (applied, _) <- evalEither (S.applied two (Applied 0 "p0" "p1" [S.digest replyA, S.digest replyB]))
    S.current applied (Current 1 "a" laterA (S.observationOf laterA) "p0") === Left (S.StateMismatch 1 "p1" "p0")
    (three, reply) <- evalEither (S.current applied (Current 1 "a" laterA (S.observationOf laterA) "p1"))
    outputs <- expected laterA firstA first' 2
    objective reply === map O.gradient outputs
    S.proximals three Map.! "a" === firstA
    (finished, closed) <- evalEither (S.applied three (Applied 1 "p1" "p2" [S.digest reply]))
    Completion.step closed === 1
    S.complete finished === Right "p2"
    S.complete three === Left S.Unfinished

scoredWith :: [Word32] -> [Word32] -> [Word32] -> Sample -> Int -> PropertyT IO [O.Output]
scoredWith references now old entry total = do
    advantages <- advantagesOf entry old
    evalEither (O.calculate profile total (zipWith5 O.Inputs now old (behaviorWords entry) references advantages))

learnerReference :: PropertyT IO ()
learnerReference = do
    let mine = [word (-1.25), word (-1.4)]
        nowA = [word (-0.9), word (-0.6)]
        laterB = [word (-1.4)]
    begun <- evalEither (S.begin exchange (planned {S.batches = [["a"], ["b"]], S.source = S.FromLearner}))
    S.current begun (Current 0 "a" nowA (S.observationOf nowA) "p0") === Left (S.MissingReference "a")
    S.reference begun "a" [word (-1)] === Left (S.LengthMismatch "a")
    withA <- evalEither (S.reference begun "a" mine)
    S.reference withA "a" mine === Left (S.DuplicateReference "a")
    withProximal <- evalEither (S.proximal withA "b" [word (-1.7)])
    (answered, reply) <- evalEither (S.current withProximal (Current 0 "a" nowA (S.observationOf nowA) "p0"))
    outputs <- scoredWith mine nowA nowA first' 2
    objective reply === map O.gradient outputs
    S.referencesOf answered Map.! "a" === mine
    later <- evalEither (S.reference answered "b" laterB)
    (stepped, _) <- evalEither (S.applied later (Applied 0 "p0" "p1" [S.digest reply]))
    (_, scoped) <- evalEither (S.current stepped (Current 1 "b" [word (-1.6)] (S.observationOf [word (-1.6)]) "p1"))
    outputsB <- scoredWith laterB [word (-1.6)] [word (-1.7)] second' 1
    objective scoped === map O.gradient outputsB
    S.referencesOf stepped Map.! "b" === laterB

outside :: PropertyT IO ()
outside = do
    begun <- evalEither (S.begin exchange (planned {S.batches = [["a"], ["b"]]}))
    let nowA = [word (-0.9), word (-0.6)]
    S.reference begun "a" [word (-1.5)] === Left (S.UnexpectedReference "a")
    S.proximal begun "a" nowA === Left (S.DuplicateProximal "a")
    S.proximal begun "b" [word (-1), word (-1)] === Left (S.LengthMismatch "b")
    (answered, reply) <- evalEither (S.current begun (Current 0 "a" nowA (S.observationOf nowA) "p0"))
    S.proximal answered "b" [word (-1.7)] === Left (S.LateProximal "b")
    S.applied answered (Applied 0 "p0" "p1" [S.digest reply]) === Left (S.MissingProximal "b")
    withProximal <- evalEither (S.proximal begun "b" [word (-1.7)])
    S.proximal withProximal "b" [word (-1.7)] === Left (S.DuplicateProximal "b")
    (answered', reply') <- evalEither (S.current withProximal (Current 0 "a" nowA (S.observationOf nowA) "p0"))
    (stepped, _) <- evalEither (S.applied answered' (Applied 0 "p0" "p1" [S.digest reply']))
    (_, later) <- evalEither (S.current stepped (Current 1 "b" [word (-1.6)] (S.observationOf [word (-1.6)]) "p1"))
    outputs <- expected [word (-1.6)] [word (-1.7)] second' 1
    objective later === map O.gradient outputs

refusals :: PropertyT IO ()
refusals = do
    begun <- evalEither (S.begin exchange planned)
    let nowA = [word (-0.9), word (-0.6)]
    S.current begun (Current 0 "b" [word (-1)] (S.observationOf [word (-1)]) "p0") === Left (S.OutOfOrder 0 "b")
    S.current begun (Current 1 "a" nowA (S.observationOf nowA) "p0") === Left (S.OutOfOrder 1 "a")
    S.current begun (Current 0 "a" nowA (S.observationOf nowA) "other") === Left (S.StateMismatch 0 "p0" "other")
    S.current begun (Current 0 "a" [word (-1)] (S.observationOf [word (-1)]) "p0") === Left (S.LengthMismatch "a")
    S.current begun (Current 0 "a" nowA "another observation" "p0") === Left (S.ObservationMismatch "a")
    S.current begun (Current 0 "a" [word 1, word (-1)] (S.observationOf [word 1, word (-1)]) "p0") === Left (S.Scalar (O.InvalidInput "Log probabilities must be nonpositive"))
    (one, replyA) <- evalEither (S.current begun (Current 0 "a" nowA (S.observationOf nowA) "p0"))
    S.applied one (Applied 0 "p0" "p1" [S.digest replyA]) === Left (S.Incomplete 0)
    (two, replyB) <- evalEither (S.current one (Current 0 "b" [word (-1.8)] (S.observationOf [word (-1.8)]) "p0"))
    S.applied two (Applied 0 "p0" "p1" [S.digest replyB, S.digest replyA]) === Left (S.ConsumedMismatch 0)
    S.applied two (Applied 0 "other" "p1" [S.digest replyA, S.digest replyB]) === Left (S.StateMismatch 0 "p0" "other")
    S.applied two (Applied 1 "p0" "p1" [S.digest replyA, S.digest replyB]) === Left (S.StepMismatch 0 1)

vector :: PropertyT IO ()
vector = S.digest (Reply 0 "a" "o" "s" [word 1, word (-0.5)] [word 0]) === "d3fde3776c065645428c5cfc2ab0cd527f3084bb494826d7601ca3241c629949"

plans :: PropertyT IO ()
plans = do
    let invalid = [([], [["a"]]), ([first', first'], [["a"]]), ([first' {S.behaviorWords = []}], [["a"]]), ([first' {S.referenceWords = [word (-1)]}], [["a"]]), ([first'], []), ([first'], [["a"], []]), ([first'], [["a", "b"]]), ([first', second'], [["a"]])]
    forM_ invalid $ \(declared, steps') -> case S.begin exchange (planned {S.members = declared, S.batches = steps'}) of
        Left (S.InvalidPlan _) -> success
        unexpected -> annotateShow (declared, steps', fmap S.identity unexpected) >> failure
    one <- evalEither (S.begin exchange planned)
    two <- evalEither (S.begin exchange (planned {S.batches = [["a"], ["b"]]}))
    other <- evalEither (S.begin exchange (planned {S.members = [first', second' {S.signal = S.Advantage (word 1)}]}))
    same <- evalEither (S.begin (V.Binding (V.CallId 9) (V.AttemptId 9) (V.Instance 9)) planned)
    learner <- evalEither (S.begin exchange (planned {S.source = S.FromLearner}))
    S.identity one === "e4876b8f1b64eeca398a5a2ef8a79df24b9da8bcc8ae3f0705fa089795e6cdb9"
    S.identity learner === "92d2c03e1ad261692db7d3d3c9aac12b41232da427f8cfb62f4d48adc0fcf05c"
    distilledPlan <- evalEither (S.begin exchange taught)
    otherTeacher <- evalEither (S.begin exchange (taught {S.algorithm = S.Distilled teacher {Definition.teacherModel = "other"}}))
    otherTemperature <- evalEither (S.begin exchange (taught {S.algorithm = S.Distilled teacher {Definition.teacherTemperature = 0.5}}))
    otherWords <- evalEither (S.begin exchange (taught {S.members = [first' {S.signal = S.Teacher [word (-0.5), word (-2)]}, second' {S.signal = S.Teacher [word (-1)]}]}))
    assert (S.identity distilledPlan `notElem` [S.identity one, S.identity otherTeacher, S.identity otherTemperature, S.identity otherWords])
    forM_ [planned {S.members = [first' {S.signal = S.Teacher [word (-1), word (-1)]}, second']}, taught {S.members = [first', second' {S.signal = S.Teacher [word (-1)]}]}, taught {S.members = [first' {S.signal = S.Teacher [word (-1)]}, second' {S.signal = S.Teacher [word (-1)]}]}] $ \mixed ->
        case S.begin exchange mixed of
            Left (S.InvalidPlan _) -> success
            unexpected -> annotateShow (fmap S.identity unexpected) >> failure
    assert (S.identity one /= S.identity two)
    assert (S.identity one /= S.identity other)
    assert (S.identity one /= S.identity learner)
    S.identity same === S.identity one

distilled :: PropertyT IO ()
distilled = do
    begun <- evalEither (S.begin exchange taught)
    let nowA = [word (-0.9), word (-0.6)]
        proximalB = [word (-1.7)]
        laterB = [word (-1.6)]
    [memberA, memberB] <- pure (S.members taught)
    withProximal <- evalEither (S.proximal begun "b" proximalB)
    (answered, replyA) <- evalEither (S.current withProximal (Current 0 "a" nowA (S.observationOf nowA) "p0"))
    outputsA <- expected nowA nowA memberA 2
    (objective replyA, surrogate replyA) === (map O.gradient outputsA, map O.surrogateGradient outputsA)
    (stepped, _) <- evalEither (S.applied answered (Applied 0 "p0" "p1" [S.digest replyA]))
    (_, replyB) <- evalEither (S.current stepped (Current 1 "b" laterB (S.observationOf laterB) "p1"))
    outputsB <- expected laterB proximalB memberB 1
    (objective replyB, surrogate replyB) === (map O.gradient outputsB, map O.surrogateGradient outputsB)
    assert (0 `notElem` surrogate replyB)
    misused <- evalEither (zipWithM O.distilled [word (-1)] laterB)
    fromCurrent <- evalEither (O.calculate profile 1 (zipWith5 O.Inputs laterB proximalB (behaviorWords memberB) (referenceWords memberB) misused))
    assert (surrogate replyB /= map O.surrogateGradient fromCurrent)
    invalid <- evalEither (S.begin exchange (taught {S.members = [first' {S.signal = S.Teacher [word 0.5, word (-2)]}, second' {S.signal = S.Teacher [word (-1)]}]}))
    fmap snd (S.current invalid (Current 0 "a" nowA (S.observationOf nowA) "p0")) === Left (S.Scalar (O.InvalidInput "Log probabilities must be nonpositive"))
