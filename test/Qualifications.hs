{-# LANGUAGE OverloadedStrings #-}

module Qualifications (qualifications) where

import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Hedgehog
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Evidence qualified as C
import Invar.Spec.Invocation qualified as I
import Invar.Spec.Load qualified as L
import Invar.Spec.Program
import Invar.Spec.Qualification qualified as Q
import Invar.Spec.Value (Scalar (..), Value (..))
import Loads qualified as Fixture

qualifications :: Group
qualifications =
    Group
        "Bound qualification and live authorization"
        [ ("subject retains the actual load and prepared operation", once actual)
        , ("unused input changes remain part of the exact subject", once inputs)
        , ("a subject requires a live matching materialization", once live)
        , ("constructing a subject grants no dispatch or completion", once inert)
        , ("checked conditional regions authorize only their actual subject", once conditional)
        , ("missing cyclic and unrelated refuted evidence cannot qualify", once unsupported)
        , ("the whole-operation goal and load cannot be assumed", once vacuous)
        , ("region conjunction cannot silently become operation refinement", once uncomposed)
        , ("revocation retains certificates but removes execution authority", once revoked)
        , ("the same emission from a different program cannot reuse qualification", once changedProgram)
        , ("empty duplicate or undeclared qualification regions are rejected", once invalidPolicy)
        ]
  where
    once = withTests 1 . property

actual :: PropertyT IO ()
actual = do
    registry <- Fixture.registered Fixture.descriptor
    ready <- Fixture.callFor (L.image Fixture.descriptor)
    selected <- evalEither (Q.subject registry Fixture.bound ready)
    loaded <- evalEither (L.historical registry (I.Instance 0))
    intended <- evalEither (I.intention ready (I.CallId 0))
    Q.actualLoad selected === loaded
    Q.invocation selected === Fixture.bound
    Q.operation selected === intended
    I.intendedCommand intended === 0
    I.intendedInputs intended === Map.empty
    I.intent ready (I.CallId 0) === Right (I.intendedEmission intended)

inputs :: PropertyT IO ()
inputs = do
    registry <- Fixture.registered Fixture.descriptor
    base <- Fixture.callFor (L.image Fixture.descriptor)
    selected <- evalEither (I.intent base (I.CallId 0))
    let source = Operational "unused"
        policyType = RecordType (Map.fromList [("artifact", SequenceType TokenType), ("profile", SequenceType TokenType)])
        kind = RecordType (Map.fromList [("policy", policyType), ("value", NumberType)])
        schema = Schema (Map.singleton source NumberType) Map.empty (Map.singleton "decode" (Sink "reference" kind Set.empty Set.empty))
        command = Emit "decode" "reference" (Constant kind (E.payload selected))
    checked <- evalEither (A.load (A.encode (E.Semantics schema Map.empty) [command]))
    let initial = I.start checked 0
        prepare value = I.prepare (I.Selection (I.CallId 0) (Map.singleton source (Atom (Number value)))) initial
    leftRuntime <- evalEither (prepare 1)
    rightRuntime <- evalEither (prepare 2)
    left <- evalEither (Q.subject registry Fixture.bound leftRuntime)
    right <- evalEither (Q.subject registry Fixture.bound rightRuntime)
    I.intendedEmission (Q.operation left) === I.intendedEmission (Q.operation right)
    assert (left /= right)
live :: PropertyT IO ()
live = do
    registry <- Fixture.registered Fixture.descriptor
    ready <- Fixture.callFor (L.image Fixture.descriptor)
    Q.subject (L.close registry) Fixture.bound ready === Left (Q.LoadError (L.Unloaded (I.Instance 0)))
    changed <- Fixture.callFor (L.Image "different-policy" "numeric-profile-0")
    case Q.subject registry Fixture.bound changed of
        Left (Q.LoadError L.DispatchImageMismatch {}) -> pure ()
        other -> annotateShow other >> failure

inert :: PropertyT IO ()
inert = do
    registry <- Fixture.registered Fixture.descriptor
    ready <- Fixture.callFor (L.image Fixture.descriptor)
    first <- evalEither (Q.subject registry Fixture.bound ready)
    second <- evalEither (Q.subject registry Fixture.bound ready)
    first === second
    I.phase ready (I.AttemptId 0) === Left (I.UnknownAttempt (I.AttemptId 0))
    liveLoad <- evalEither (L.acquire registry (I.Instance 0))
    issued <- evalEither (L.dispatch (L.Dispatch liveLoad Fixture.bound) registry ready)
    Q.subject registry Fixture.bound issued === Left (Q.LoadError (L.InvocationError (I.CallInUse (I.CallId 0) (I.AttemptId 0))))

definition :: Q.Definition
definition = Q.Definition "native-forward-correspondence/v1" "pinned-executable-reference" "actual output words"

targetDefinition :: Q.Definition
targetDefinition = Q.Definition "native-operation-refinement/v1" "pinned-operation-reference" "complete operation observation"

qualificationKey :: Q.Key
qualificationKey = Q.Key 0

fixture :: PropertyT IO (Q.Registry, L.Registry, I.Runtime, Q.Request)
fixture = do
    declared <- evalEither (Q.policy targetDefinition [definition] (Q.Premises [definition] True))
    authority <- evalEither (Q.register qualificationKey declared Q.empty)
    registry <- Fixture.registered Fixture.descriptor
    ready <- Fixture.callFor (L.image Fixture.descriptor)
    selected <- evalEither (Q.subject registry Fixture.bound ready)
    requested <- evalEither (Q.request authority qualificationKey selected)
    pure (authority, registry, ready, requested)

derivation :: Q.Request -> (C.Graph, C.EvidenceId)
derivation requested = (Map.fromList (nodes ++ composition), root)
  where
    expected = Q.conclusion requested
    required = Q.requiredRegions requested
    claims = Q.regionClaims requested
    rule C.OutputEqual {} = C.Compare
    rule _ = C.Assume
    nodes = [(C.EvidenceId index, C.Node claim (rule claim)) | (index, claim) <- zip [0 ..] claims]
    count = fromIntegral (length nodes)
    regions = C.EvidenceId count
    law = C.EvidenceId (count + 1)
    root = C.EvidenceId (count + 2)
    composition =
        [ (regions, C.Node required (C.Conjoin (map fst nodes)))
        , (law, C.Node (Q.composition requested) C.Assume)
        , (root, C.Node expected (C.Apply law regions))
        ]

qualified :: Q.Request -> PropertyT IO Q.QualifiedResult
qualified requested = let (graph, root) = derivation requested in evalEither (Q.qualify requested graph root)

conditional :: PropertyT IO ()
conditional = do
    (authority, registry, ready, requested) <- fixture
    accepted <- qualified requested
    C.conclusion (Q.certificate accepted) === Q.conclusion requested
    C.assumptions (Q.certificate accepted) === Q.permittedAssumptions requested
    assert (C.ReportComparison `elem` C.methods (Q.certificate accepted))
    assert (C.Hypothesis `elem` C.methods (Q.certificate accepted))
    assert (C.ImplicationElimination `elem` C.methods (Q.certificate accepted))
    issued <- evalEither (Q.dispatch (authority, registry) accepted ready)
    I.phase issued (I.AttemptId 0) === Right I.Issued
    selected <- evalEither (Q.subject registry Fixture.bound ready)
    Q.qualifiedSubject accepted === selected

unsupported :: PropertyT IO ()
unsupported = do
    (_, _, _, requested) <- fixture
    let root = C.EvidenceId 0
        expected = Q.conclusion requested
    Q.qualify requested Map.empty root === Left (Q.Inconclusive (C.MissingEvidence root))
    Q.qualify requested (Map.singleton root (C.Node expected (C.Discharge root []))) root === Left (Q.Inconclusive (C.Cycle root))
    selected <- qualified requested
    let report = L.report (Q.actualLoad (Q.qualifiedSubject selected))
        foreignClaim = C.OutputEqual report "different-output"
    Q.qualify requested (Map.singleton root (C.Node foreignClaim C.Compare)) root === Left (Q.WrongConclusion expected foreignClaim)

vacuous :: PropertyT IO ()
vacuous = do
    (authority, registry, ready, requested) <- fixture
    let (graph, root) = derivation requested
        expected = Q.conclusion requested
    Q.qualify requested (Map.singleton root (C.Node expected C.Assume)) root === Left (Q.UnapprovedHypothesis expected)
    let loadNode = C.EvidenceId 0
    case Map.lookup loadNode graph of
        Just (C.Node observed _) -> Q.qualify requested (Map.insert loadNode (C.Node observed C.Assume) graph) root === Left (Q.UnapprovedHypothesis observed)
        Nothing -> failure
    declared <- evalEither (Q.policy targetDefinition [definition] (Q.Premises [] True))
    strict <- evalEither (Q.register (Q.Key 1) declared authority)
    selected <- evalEither (Q.subject registry Fixture.bound ready)
    required <- evalEither (Q.request strict (Q.Key 1) selected)
    let (unproved, requestedRoot) = derivation required
    case Q.qualify required unproved requestedRoot of
        Left Q.UnapprovedHypothesis {} -> pure ()
        other -> annotateShow other >> failure

revoked :: PropertyT IO ()
revoked = do
    (authority, registry, ready, requested) <- fixture
    accepted <- qualified requested
    let certificate = Q.certificate accepted
    retired <- evalEither (Q.revoke qualificationKey authority)
    rejected (Q.Revoked qualificationKey) (Q.dispatch (retired, registry) accepted ready)
    rejected (Q.Revoked qualificationKey) (Q.dispatch (Q.close authority, registry) accepted ready)
    rejected (Q.LoadError (L.Unloaded (I.Instance 0))) (Q.dispatch (authority, L.close registry) accepted ready)
    declared <- evalEither (Q.policy targetDefinition [definition] (Q.Premises [definition] True))
    rejected (Q.DuplicateKey qualificationKey) (Q.register qualificationKey declared retired)
    Q.certificate accepted === certificate
    C.assumptions certificate === Q.permittedAssumptions requested

changedProgram :: PropertyT IO ()
changedProgram = do
    (authority, registry, ready, requested) <- fixture
    accepted <- qualified requested
    selected <- evalEither (I.intent ready (I.CallId 0))
    let policyType = RecordType (Map.fromList [("artifact", SequenceType TokenType), ("profile", SequenceType TokenType)])
        kind = RecordType (Map.fromList [("policy", policyType), ("value", NumberType)])
        schema = Schema Map.empty Map.empty (Map.singleton "decode" (Sink "reference" kind Set.empty Set.empty))
        command = Emit "decode" "reference" (Constant kind (E.payload selected))
    checked <- evalEither (A.load (A.encode (E.Semantics schema Map.empty) [command, command]))
    changed <- evalEither (I.prepare (I.Selection (I.CallId 0) Map.empty) (I.start checked 1))
    I.intent changed (I.CallId 0) === Right selected
    rejected Q.SubjectMismatch (Q.dispatch (authority, registry) accepted changed)

rejected :: Q.Error -> Either Q.Error value -> PropertyT IO ()
rejected expected result = case result of
    Left problem -> problem === expected
    Right _ -> failure

invalidPolicy :: PropertyT IO ()
invalidPolicy = do
    Q.policy targetDefinition [] (Q.Premises [] False) === Left (Q.MalformedPolicy "Qualification requires a nonempty region inventory")
    Q.policy targetDefinition [definition, definition] (Q.Premises [] False) === Left (Q.MalformedPolicy "Qualification predicates must be distinct")
    Q.policy targetDefinition [definition] (Q.Premises [definition {Q.reference = "different-reference"}] True) === Left (Q.MalformedPolicy "Declared assumptions must be distinct exact required regions")
    Q.policy definition [definition] (Q.Premises [definition] True) === Left (Q.MalformedPolicy "The whole-operation predicate must differ from its primitive regions")

uncomposed :: PropertyT IO ()
uncomposed = do
    (authority, registry, ready, requested) <- fixture
    let (graph, _) = derivation requested
        regions = C.EvidenceId 2
    Q.qualify requested graph regions === Left (Q.WrongConclusion (Q.conclusion requested) (Q.requiredRegions requested))
    declared <- evalEither (Q.policy targetDefinition [definition] (Q.Premises [definition] False))
    strict <- evalEither (Q.register (Q.Key 1) declared authority)
    selected <- evalEither (Q.subject registry Fixture.bound ready)
    required <- evalEither (Q.request strict (Q.Key 1) selected)
    let (unproved, root) = derivation required
    Q.qualify required unproved root === Left (Q.UnapprovedHypothesis (Q.composition required))
