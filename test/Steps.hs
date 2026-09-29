{-# LANGUAGE OverloadedStrings #-}

module Steps (steps) where

import Data.Map.Strict qualified as Map
import Data.Word (Word32)
import GHC.Float (castFloatToWord32)
import Hedgehog
import Invar.Learn.Objective qualified as O
import Invar.Learn.Stream (Applied (..), Current (..), Reply (..), Sample (..))
import Invar.Learn.Stream qualified as S

steps :: Group
steps =
    Group
        "Per-step learner exchange"
        [ ("one full step reuses each first observation as proximal and answers with the core's cotangents", once full)
        , ("later steps keep proximal fixed and chain the learner state", once staged)
        , ("samples outside the first step need proximal before the first update", once outside)
        , ("reports out of order, from another state or with other cotangents are refused", once refusals)
        , ("the cotangent digest is SHA-256 over little-endian FP32 words", once vector)
        ]
  where
    once = withTests 1 . property

word :: Float -> Word32
word = castFloatToWord32

profile :: O.Profile
profile = O.Profile 0.2 0.04

first', second' :: Sample
first' = Sample "a" [word (-1), word (-0.5)] [] (word 1)
second' = Sample "b" [word (-2)] [word (-1.5)] (word (-1))

expected :: [Word32] -> [Word32] -> Sample -> Int -> PropertyT IO [O.Output]
expected now old entry total = evalEither (O.calculate profile total [O.Inputs c p b r (advantageWord entry) | (c, p, b, r) <- zip4 now old (behaviorWords entry) references])
  where
    references = if null (referenceWords entry) then behaviorWords entry else referenceWords entry
    zip4 (a : as) (b : bs) (c : cs) (d : ds) = (a, b, c, d) : zip4 as bs cs ds
    zip4 _ _ _ _ = []

full :: PropertyT IO ()
full = do
    let begun = S.begin profile "p0" [first', second'] [["a", "b"]]
        nowA = [word (-0.9), word (-0.6)]
        nowB = [word (-1.8)]
    (afterA, replyA) <- evalEither (S.current begun (Current 0 "a" nowA (S.observationOf nowA) "p0"))
    outputsA <- expected nowA nowA first' 3
    objective replyA === map O.gradient outputsA
    reward replyA === map O.rewardGradient outputsA
    (afterB, replyB) <- evalEither (S.current afterA (Current 0 "b" nowB (S.observationOf nowB) "p0"))
    outputsB <- expected nowB nowB second' 3
    objective replyB === map O.gradient outputsB
    S.proximals afterB === Map.fromList [("a", nowA), ("b", nowB)]
    done <- evalEither (S.applied afterB (Applied 0 "p0" "p1" [S.digest replyA, S.digest replyB]))
    S.complete done === Right "p1"
    S.current done (Current 1 "a" nowA (S.observationOf nowA) "p1") === Left S.Finished

staged :: PropertyT IO ()
staged = do
    let begun = S.begin profile "p0" [first', second'] [["a", "b"], ["a"]]
        firstA = [word (-0.9), word (-0.6)]
        laterA = [word (-0.7), word (-0.8)]
    (one, replyA) <- evalEither (S.current begun (Current 0 "a" firstA (S.observationOf firstA) "p0"))
    (two, replyB) <- evalEither (S.current one (Current 0 "b" [word (-1.8)] (S.observationOf [word (-1.8)]) "p0"))
    applied <- evalEither (S.applied two (Applied 0 "p0" "p1" [S.digest replyA, S.digest replyB]))
    S.current applied (Current 1 "a" laterA (S.observationOf laterA) "p0") === Left (S.StateMismatch 1 "p1" "p0")
    (three, reply) <- evalEither (S.current applied (Current 1 "a" laterA (S.observationOf laterA) "p1"))
    outputs <- expected laterA firstA first' 2
    objective reply === map O.gradient outputs
    S.proximals three Map.! "a" === firstA
    finished <- evalEither (S.applied three (Applied 1 "p1" "p2" [S.digest reply]))
    S.complete finished === Right "p2"
    S.complete three === Left S.Unfinished

outside :: PropertyT IO ()
outside = do
    let begun = S.begin profile "p0" [first', second'] [["a"], ["b"]]
        nowA = [word (-0.9), word (-0.6)]
    S.proximal begun "a" nowA === Left (S.DuplicateProximal "a")
    S.proximal begun "b" [word (-1), word (-1)] === Left (S.LengthMismatch "b")
    (answered, reply) <- evalEither (S.current begun (Current 0 "a" nowA (S.observationOf nowA) "p0"))
    S.proximal answered "b" [word (-1.7)] === Left (S.LateProximal "b")
    S.applied answered (Applied 0 "p0" "p1" [S.digest reply]) === Left (S.MissingProximal "b")
    withProximal <- evalEither (S.proximal begun "b" [word (-1.7)])
    S.proximal withProximal "b" [word (-1.7)] === Left (S.DuplicateProximal "b")
    (answered', reply') <- evalEither (S.current withProximal (Current 0 "a" nowA (S.observationOf nowA) "p0"))
    stepped <- evalEither (S.applied answered' (Applied 0 "p0" "p1" [S.digest reply']))
    (_, later) <- evalEither (S.current stepped (Current 1 "b" [word (-1.6)] (S.observationOf [word (-1.6)]) "p1"))
    outputs <- expected [word (-1.6)] [word (-1.7)] second' 1
    objective later === map O.gradient outputs

refusals :: PropertyT IO ()
refusals = do
    let begun = S.begin profile "p0" [first', second'] [["a", "b"]]
        nowA = [word (-0.9), word (-0.6)]
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
