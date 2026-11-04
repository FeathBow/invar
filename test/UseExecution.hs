{-# LANGUAGE OverloadedStrings #-}

module UseExecution (useExecution) where

import Data.List (sort)
import Data.Map.Strict qualified as Map
import Data.Ratio ((%))
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Numerical qualified as N
import Invar.Use qualified as U
import Invar.Use.Execution qualified as E
import UseAdmission (contractFor)
import UseFixture qualified as F

useExecution :: Group
useExecution =
    Group
        "Declared execution arrangements"
        [ ("every arrangement runs each input exactly once in bounded groups", withTests 200 (property partition))
        , ("a repeat must change the request order or the batch partition", withTests 1 (property schedules))
        , ("a scored or probed requirement needs its collection entry and scorers", once collectionCoverage)
        , ("probe collections reject conflicting or Hugging Face probe plans", once probeRules)
        ]
  where
    once = withTests 1 . property

partition :: PropertyT IO ()
partition = do
    count <- forAll (Gen.integral (Range.linear 0 40))
    size <- forAll (Gen.integral (Range.linear 1 12))
    selected <- forAll (Gen.element [E.Declared, E.Reversed, E.Rotated 3, E.Rotated 41])
    let groups = E.arrange (E.Arrangement selected size) [0 .. count - 1 :: Int]
    sort (concat groups) === [0 .. count - 1]
    assert (all (\group -> not (null group) && length group <= fromIntegral size) groups)

schedules :: PropertyT IO ()
schedules = do
    let paired = E.Arrangement E.Declared 8
    E.unchanged 16 paired [E.Arrangement E.Declared 8] === [0]
    E.unchanged 16 paired [E.Arrangement (E.Rotated 16) 8] === [0]
    E.unchanged 16 paired [E.Arrangement E.Reversed 16, E.Arrangement E.Declared 16] === []
    E.unchanged 16 paired [E.Arrangement E.Reversed 8, E.Arrangement E.Reversed 8] === [0, 1]
    E.unchanged 16 paired [E.Arrangement E.Reversed 8, E.Arrangement (E.Rotated 3) 5] === []

collectionCoverage :: PropertyT IO ()
collectionCoverage = do
    supplied <- F.fixture F.trials
    base <- contractFor supplied
    let scored = base {U.criterion = (U.criterion base) {U.numericalRequirements = [U.NumericalRequirement (N.ScoredPathLogRatioWithin N.Candidate (1 % 2)) [] "score only"]}}
        scorer = E.Scorer "vllm" "worker" "config" "adapter"
        complete = E.Collection [N.Candidate] [] (Map.fromList [(N.Reference, scorer), (N.Candidate, scorer)])
    E.collectionProblems scored E.emptyCollection /== []
    E.collectionProblems scored complete === []
    E.collectionProblems scored (E.Collection [N.Candidate] [] Map.empty) /== []
    E.collectionProblems scored (E.Collection [N.Reference, N.Candidate] [] (Map.fromList [(N.Reference, scorer), (N.Candidate, scorer)])) /== []
    let probed = base {U.criterion = (U.criterion base) {U.numericalRequirements = [U.NumericalRequirement (N.FullVocabularyKLWithin N.Candidate N.ReferenceToCandidate (1 % 2)) [0, 1] "probe only"]}}
        probedCollection = E.Collection [] [E.ProbePlan N.Candidate [0, 1]] (Map.fromList [(N.Reference, scorer), (N.Candidate, scorer)])
    E.collectionProblems probed probedCollection === []
    E.collectionProblems probed (probedCollection {E.probes = []}) /== []
    E.collectionProblems probed (probedCollection {E.probes = [E.ProbePlan N.Candidate [0, 1], E.ProbePlan N.Candidate [0, 1]]}) /== []

probeRules :: PropertyT IO ()
probeRules = do
    supplied <- F.fixture F.trials
    base <- contractFor supplied
    let scorer = E.Scorer "vllm" "worker" "config" "adapter"
        requirement steps = U.NumericalRequirement (N.FullVocabularyKLWithin N.Candidate N.ReferenceToCandidate (1 % 2)) steps "probe"
        contract = base {U.criterion = (U.criterion base) {U.numericalRequirements = [requirement [0, 1]]}}
        collection steps = E.Collection [] [E.ProbePlan N.Candidate steps] (Map.fromList [(N.Reference, scorer), (N.Candidate, scorer)])
    E.collectionProblems contract (collection [0, 2]) /== []
    let hugging = E.Collection [] [E.ProbePlan N.Candidate [0, 1]] (Map.fromList [(N.Reference, scorer {E.scorerBackend = "hf"}), (N.Candidate, scorer {E.scorerBackend = "hf"})])
    assert (any ((== "The Hugging Face scorer records no full-vocabulary probe") . E.collectionMessage) (E.collectionProblems contract hugging))
    let conflicting = base {U.criterion = (U.criterion base) {U.numericalRequirements = [requirement [0], requirement [1]]}}
    E.collectionProblems conflicting (collection [0]) /== []
