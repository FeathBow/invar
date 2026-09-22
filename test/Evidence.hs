{-# LANGUAGE OverloadedStrings #-}

module Evidence (evidence) where

import Control.Monad (forM_)
import Data.ByteString (ByteString)
import Data.List ((\\))
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Evidence qualified as C
import Invar.Spec.Invocation qualified as I
import Invar.Spec.Program
import Invar.Spec.Value (Scalar (..), Value (..))
import Properties (campaign)

evidence :: Group
evidence =
    Group
        "Evidence judgements"
        [ ("report equality produces only its exact claim", once comparison)
        , ("unequal report bytes retain a concrete witness", once counterexample)
        , ("external claims remain explicit hypotheses", once hypotheses)
        , ("discharge removes only the proved hypothesis", once discharge)
        , ("self-support cannot erase an assumption", once selfSupport)
        , ("every obligation dimension participates in matching", once dimensions)
        , ("missing and cyclic evidence remain unknown", once invalidGraphs)
        , ("unreachable invalid nodes do not change the root judgement", once localGraphs)
        , ("local refutation lifts only to the declared conjunction", once lifting)
        , ("shared proof nodes preserve assumptions", once shared)
        , ("conjunction permutations preserve verdicts and certificate meaning", campaign conjunctionPermutations)
        , ("implication application retains its exact carried premises", once application)
        , ("application requires the declared antecedent and consequent", once applicationBindings)
        , ("a false premise does not refute an implication's conclusion", once falsePremise)
        ]
  where
    once = withTests 1 . property

first, second, third, fourth :: C.EvidenceId
first = C.EvidenceId 0
second = C.EvidenceId 1
third = C.EvidenceId 2
fourth = C.EvidenceId 3

obligation :: C.Obligation
obligation = C.Obligation "refines" "reference-v1" "token-bits" "one-case" "load-1"

external :: C.Claim
external = C.External obligation

completed :: ByteString -> PropertyT IO I.Completion
completed output = do
    let value = Atom (Number 3)
        schema = Schema Map.empty Map.empty (Map.singleton "out" (Sink "reference" NumberType Set.empty Set.empty))
    checked <- evalEither (A.load (A.encode (E.Semantics schema Map.empty) [Emit "out" "reference" (Constant NumberType value)]))
    let bound = I.Binding (I.CallId 0) (I.AttemptId 0) (I.Instance 0)
    prepared <- evalEither (I.prepare (I.Selection (I.CallId 0) Map.empty) (I.start checked 0))
    sent <- evalEither (I.issue bound prepared)
    used <- evalEither (I.consume (I.Consumption bound (A.bytes checked) (E.Emission "out" "reference" value)) sent)
    done <- evalEither (I.finish bound output used)
    evalEither (I.completion done (I.AttemptId 0)) >>= evalMaybe

accepted :: C.Verdict -> PropertyT IO C.Certificate
accepted (C.Accept certificate) = pure certificate
accepted other = annotateShow other >> failure

refuted :: C.Verdict -> PropertyT IO C.Counterexample
refuted (C.Refute result) = pure result
refuted other = annotateShow other >> failure

comparison :: PropertyT IO ()
comparison = do
    report <- completed "actual"
    let goal = C.OutputEqual report "actual"
    result <- accepted (C.check (Map.singleton first (C.Node goal C.Compare)) first)
    C.conclusion result === goal
    C.assumptions result === []
    C.methods result === [C.ReportComparison]
    C.check (Map.singleton first (C.Node external C.Compare)) first === C.Unknown (C.UnsupportedComparison external)

counterexample :: PropertyT IO ()
counterexample = do
    report <- completed "actual"
    let goal = C.OutputEqual report "expected"
    result <- refuted (C.check (Map.singleton first (C.Node goal C.Compare)) first)
    C.refuted result === goal
    C.witness result === C.OutputWitness report "expected"
    assert (I.completedOutput report /= "expected")

hypotheses :: PropertyT IO ()
hypotheses = do
    result <- accepted (C.check (Map.singleton first (C.Node external C.Assume)) first)
    C.conclusion result === external
    C.assumptions result === [external]
    C.methods result === [C.Hypothesis]

discharge :: PropertyT IO ()
discharge = do
    report <- completed "actual"
    let equality = C.OutputEqual report "actual"
        composite = C.All [external, equality]
        graph =
            Map.fromList
                [ (first, C.Node external C.Assume)
                , (second, C.Node equality C.Assume)
                , (third, C.Node composite (C.Conjoin [first, second]))
                , (fourth, C.Node equality C.Compare)
                , (C.EvidenceId 4, C.Node composite (C.Discharge third [fourth]))
                ]
    result <- accepted (C.check graph (C.EvidenceId 4))
    C.conclusion result === composite
    C.assumptions result === [external]
    assert (C.ReportComparison `elem` C.methods result)
    assert (C.AssumptionDischarge `elem` C.methods result)

selfSupport :: PropertyT IO ()
selfSupport = do
    let graph = Map.fromList [(first, C.Node external C.Assume), (second, C.Node external (C.Discharge first [first]))]
    result <- accepted (C.check graph second)
    C.assumptions result === [external]

dimensions :: PropertyT IO ()
dimensions = forM_ changed $ \different -> do
    let other = C.External different
        graph = Map.fromList [(first, C.Node external C.Assume), (second, C.Node other C.Assume), (third, C.Node external (C.Discharge first [second]))]
    C.check graph third === C.Unknown (C.UnmatchedDischarge other)
  where
    changed =
        [ obligation {C.predicate = "preserves"}
        , obligation {C.specification = "reference-v2"}
        , obligation {C.observation = "logprob-bits"}
        , obligation {C.domain = "all-cases"}
        , obligation {C.binding = "load-2"}
        ]

invalidGraphs :: PropertyT IO ()
invalidGraphs = do
    C.check Map.empty first === C.Unknown (C.MissingEvidence first)
    let cycleGraph = Map.fromList [(first, C.Node external (C.Discharge second [])), (second, C.Node external (C.Discharge first []))]
    C.check cycleGraph first === C.Unknown (C.Cycle first)
    let missing = Map.singleton first (C.Node external (C.Discharge second []))
    C.check missing first === C.Unknown (C.MissingEvidence second)
    report <- completed "actual"
    let falseClaim = C.OutputEqual report "other"
        withFalse = Map.insert third (C.Node falseClaim C.Compare) cycleGraph
        whole = C.All [external, falseClaim]
    C.check (Map.insert fourth (C.Node whole (C.Conjoin [first, third])) withFalse) fourth === C.Unknown (C.Cycle first)

localGraphs :: PropertyT IO ()
localGraphs = do
    let root = Map.singleton first (C.Node external C.Assume)
        graph = Map.insert second (C.Node external (C.Discharge second [])) (Map.insert third (C.Node external (C.Discharge fourth [])) root)
    C.check graph first === C.check root first
    result <- accepted (C.check graph first)
    C.assumptions result === [external]
    C.check graph second === C.Unknown (C.Cycle second)
    C.check graph third === C.Unknown (C.MissingEvidence fourth)
    report <- completed "actual"
    let refutation = C.Node (C.OutputEqual report "other") C.Compare
    C.check (Map.insert first refutation graph) first === C.check (Map.singleton first refutation) first

lifting :: PropertyT IO ()
lifting = do
    report <- completed "actual"
    let falseClaim = C.OutputEqual report "other"
        base = Map.singleton first (C.Node falseClaim C.Compare)
        conjunction = C.All [falseClaim]
    C.check (Map.insert second (C.Node external (C.Conjoin [first])) base) second === C.Unknown (C.WrongConclusion conjunction external)
    result <- refuted (C.check (Map.insert second (C.Node conjunction (C.Conjoin [first])) base) second)
    C.refuted result === conjunction
    C.witness result === C.OutputWitness report "other"

shared :: PropertyT IO ()
shared = do
    let pair = C.All [external, external]
        graph = Map.fromList [(first, C.Node external C.Assume), (second, C.Node pair (C.Conjoin [first, first])), (third, C.Node (C.All [pair, pair]) (C.Conjoin [second, second]))]
    result <- accepted (C.check graph third)
    C.assumptions result === [external]
    empty <- accepted (C.check (Map.singleton first (C.Node (C.All []) (C.Conjoin []))) first)
    C.conclusion empty === C.All []
    C.assumptions empty === []

conjunctionPermutations :: PropertyT IO ()
conjunctionPermutations = do
    report <- completed "actual"
    let different = C.External obligation {C.observation = "another-observation"}
        equal = C.OutputEqual report "actual"
        pair = C.All [external, equal]
        graph =
            Map.fromList
                [ (first, C.Node external C.Assume)
                , (second, C.Node different C.Assume)
                , (third, C.Node equal C.Compare)
                , (fourth, C.Node pair (C.Conjoin [first, third]))
                , (C.EvidenceId 4, C.Node (C.OutputEqual report "other") C.Compare)
                , (C.EvidenceId 5, C.Node (C.OutputEqual report "different") C.Compare)
                , (C.EvidenceId 6, C.Node external C.Compare)
                , (C.EvidenceId 7, C.Node different C.Compare)
                ]
        acceptedNames = [first, second, third, fourth]
        refutations = map C.EvidenceId [4, 5]
        unknowns = map C.EvidenceId [6, 7]
        root = C.EvidenceId 8
        judge names = do
            nodes <- traverse (evalMaybe . (`Map.lookup` graph)) names
            let target = C.All (map C.claim nodes)
            pure (C.check (Map.insert root (C.Node target (C.Conjoin names)) graph) root)
    repeated <- forAll (Gen.list (Range.linear 0 8) (Gen.element acceptedNames))
    let names = acceptedNames ++ repeated
    forM_ [[], names, names ++ refutations, names ++ unknowns, names ++ refutations ++ unknowns] $ \original -> do
        reordered <- forAll (Gen.shuffle original)
        before <- judge original
        after <- judge reordered
        sameConjunctionVerdict before after

-- All preserves multiplicity; certificate premises and methods have set meaning.
-- First diagnostics and counterexample witnesses may change with traversal order.
sameConjunctionVerdict :: C.Verdict -> C.Verdict -> PropertyT IO ()
sameConjunctionVerdict (C.Accept before) (C.Accept after) = do
    sameConjunction (C.conclusion before) (C.conclusion after)
    assert (sameSet (C.assumptions before) (C.assumptions after))
    assert (sameSet (C.methods before) (C.methods after))
sameConjunctionVerdict (C.Refute before) (C.Refute after) = sameConjunction (C.refuted before) (C.refuted after)
sameConjunctionVerdict (C.Unknown _) (C.Unknown _) = pure ()
sameConjunctionVerdict before after = annotateShow (before, after) >> failure

sameConjunction :: C.Claim -> C.Claim -> PropertyT IO ()
sameConjunction (C.All before) (C.All after) = do
    before \\ after === []
    after \\ before === []
sameConjunction before after = annotateShow (before, after) >> failure

sameSet :: (Eq value) => [value] -> [value] -> Bool
sameSet before after = all (`elem` after) before && all (`elem` before) after

application :: PropertyT IO ()
application = do
    report <- completed "actual"
    let equal = C.OutputEqual report "actual"
        implication = C.Implies equal external
        graph =
            Map.fromList
                [ (first, C.Node equal C.Compare)
                , (second, C.Node implication C.Assume)
                , (third, C.Node external (C.Apply second first))
                ]
    result <- accepted (C.check graph third)
    C.conclusion result === external
    C.assumptions result === [implication]
    assert (C.ImplicationElimination `elem` C.methods result)
    assert (C.ReportComparison `elem` C.methods result)

applicationBindings :: PropertyT IO ()
applicationBindings = do
    let different = C.External obligation {C.observation = "different-observation"}
        implication = C.Implies external different
        base = Map.fromList [(first, C.Node external C.Assume), (second, C.Node implication C.Assume)]
    C.check (Map.insert third (C.Node external (C.Apply second first)) base) third === C.Unknown (C.WrongConclusion different external)
    let mismatch = Map.insert first (C.Node different C.Assume) base
    C.check (Map.insert third (C.Node different (C.Apply second first)) mismatch) third === C.Unknown (C.WrongConclusion external different)
    C.check (Map.insert third (C.Node different (C.Apply first first)) base) third === C.Unknown (C.UnsupportedApplication external)

falsePremise :: PropertyT IO ()
falsePremise = do
    report <- completed "actual"
    let unequal = C.OutputEqual report "different"
        implication = C.Implies unequal external
        graph =
            Map.fromList
                [ (first, C.Node unequal C.Compare)
                , (second, C.Node implication C.Assume)
                , (third, C.Node external (C.Apply second first))
                ]
    C.check graph third === C.Unknown (C.RefutedPremise unequal)
