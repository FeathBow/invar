{-# LANGUAGE OverloadedStrings #-}

module UseAdmission (useAdmission, contractFor) where

import Calls (change, field, request, wire)
import Control.Monad (forM_)
import Data.Aeson (Value (..), eitherDecodeStrict, encode, object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as KeyMap
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as Lines
import Data.ByteString.Lazy qualified as Lazy
import Data.Either (isLeft)
import Data.List.NonEmpty qualified as NonEmpty
import Data.Ratio ((%))
import Data.Text qualified as Text
import Data.Word (Word32)
import Hedgehog
import Invar.Infer qualified as Infer
import Invar.Numerical qualified as N
import Invar.Policy qualified as Policy
import Invar.Spec.Evidence qualified as E
import Invar.Use qualified as U
import Numeric.Natural (Natural)
import UseFixture qualified as F

useAdmission :: Group
useAdmission =
    Group
        "Explicit use admission"
        [ ("joint finite evidence retains its declared external reliance", once finite)
        , ("a small difference cannot admit an inadequate reference", once inadequate)
        , ("missing numerical evidence stays unknown despite task success", once missingNumerical)
        , ("task success cannot override a numerical counterexample", once numericalViolation)
        , ("a single metric or finite evidence cannot fulfill another standard", once strength)
        , ("workload implementation and context must match the input contract", once domain)
        , ("uncovered external premises do not grant admission", once reliance)
        , ("invalid contracts and unsupported derivations stay explicit", once invalid)
        , ("Hoeffding uses 32 prompts rather than 128 seed outcomes", once statistical)
        , ("structured contracts preserve exact requirements and reject ambiguous inputs", once encoding)
        , ("repeated candidate executions certify only the declared invariance relation", once invariance)
        , ("a diverging repeat refutes invariance with that pair as the witness", once divergentRepeat)
        , ("invariance contracts need enough executions, exact relations and declared schedule variation", once invarianceContract)
        , ("a declared transfer carries evidence only under a relied-on preservation premise", once transfer)
        ]
  where
    once = withTests 1 . property

contractFor :: U.BoundRun -> PropertyT IO U.UseContract
contractFor supplied = do
    implementation <- evalEither (Policy.describe ("test-model", "test-revision") (Infer.artifact request, Infer.tokenizer request, Infer.base request, Infer.assembly request))
    pure
        U.UseContract
            { U.purpose = "Finite test fixture mechanism only"
            , U.declaredDomain = U.domain supplied
            , U.declaredMeasurement = U.measurement supplied
            , U.referenceImplementation = implementation
            , U.candidateImplementation = implementation
            , U.maximumContext = 3
            , U.criterion =
                U.Criterion
                    [U.NumericalRequirement N.SameTermination [] "Fixture termination comparison"]
                    (Just (U.LossRequirement (U.Budget (1 % 2) "Exact fixture boundary") (U.Budget (1 % 6) "Exact fixture boundary") U.FiniteDomain))
                    []
            , U.freezeProtocol = "Fixture declaration only; not an actual sealed execution"
            , U.isolationProtocol = "Fixture-only access declaration"
            , U.selectionProtocol = "One declared fixture comparison"
            , U.reliance = [U.Reliance kind "Test-only named authority" (Bytes.replicate 32 7) | kind <- [minBound .. maxBound]]
            , U.transfers = []
            }

established :: U.UseContract -> U.Observed -> U.Finding
established contract observed = U.establish (U.Required (U.scope observed) (U.criterion contract)) observed

accepted :: U.Decision -> PropertyT IO U.Admission
accepted (U.Admitted result) = pure result
accepted other = annotateShow other >> failure

unknown :: U.Decision -> PropertyT IO [U.AdmissionProblem]
unknown (U.Undetermined reasons) = pure reasons
unknown other = annotateShow other >> failure

finite :: PropertyT IO ()
finite = do
    supplied <- F.fixture F.trials
    contract <- contractFor supplied
    observed <- evalEither (U.observe supplied)
    let result = U.admit contract (established contract observed)
    admitted <- accepted result
    U.admissionContract admitted === contract
    U.admissionScope admitted === U.scope observed
    length (U.conditions admitted) === 57
    length (E.assumptions (U.evidence admitted)) === 53
    field "status" (U.describeDecision result) === String "admitted_under_declared_reliance"
    assert (all ((== 32) . Bytes.length . U.basis . U.supporting) (U.conditions admitted))

inadequate :: PropertyT IO ()
inadequate = do
    supplied <- F.fixture [trial {F.before = "#### 99", F.after = "#### 99"} | trial <- F.trials]
    contract <- contractFor supplied
    observed <- evalEither (U.observe supplied)
    U.mean U.LossIncrease observed === Just 0
    let result = U.admit contract (established contract observed)
    case result of
        U.Rejected counterexample -> do
            E.witness counterexample === E.TaskLossWitness observed
            field "witness" (field "finding" (U.describeDecision result)) === U.describe observed
        other -> annotateShow other >> failure

numericalViolation :: PropertyT IO ()
numericalViolation = do
    supplied <- F.fixture F.trials
    contract <- contractFor supplied
    changed <- case U.cases supplied of
        [] -> failure
        first : remaining -> do
            let pair = U.paired first
                candidate = N.candidate pair
            events <- evalEither (traverse eitherDecodeStrict (Lines.lines (N.logBytes candidate)))
            let alter event
                    | field "stage" event == String "result" = change "tokens" (toJSON [1, 2, 4 :: Int]) event
                    | otherwise = event
                modified = candidate {N.logBytes = wire (map alter events)}
            pure supplied {U.cases = first {U.paired = pair {N.candidate = modified}} : remaining}
    observed <- evalEither (U.observe changed)
    let criterion = (U.criterion contract) {U.numericalRequirements = [U.NumericalRequirement N.SameTokens [] "Exact fixture comparison"]}
        requested = contract {U.criterion = criterion}
        result = U.admit requested (established requested observed)
    U.mean U.LossIncrease observed === Just (1 % 6)
    case result of
        U.Rejected counterexample -> case E.witness counterexample of
            E.NumericalWitness measured -> field "witness" (field "finding" (U.describeDecision result)) === N.describe measured
            other -> annotateShow other >> failure
        other -> annotateShow other >> failure

missingNumerical :: PropertyT IO ()
missingNumerical = do
    supplied <- F.fixture F.trials
    contract <- contractFor supplied
    observed <- evalEither (U.observe supplied)
    let criterion = (U.criterion contract) {U.numericalRequirements = [U.NumericalRequirement (N.FullVocabularyKLWithin N.Reference N.ReferenceToCandidate 1) [0] "Preselected fixture probe"]}
        requested = contract {U.criterion = criterion}
    reasons <- unknown (U.admit requested (established requested observed))
    reasons === [U.FindingUnknown (E.NumericalProblem N.MissingFullVocabulary)]

population :: U.Population
population = U.Population "Fixture population only" "Independent prompt draws are an explicit premise" "Four predeclared seeds per prompt" (1 % 40) (1 % 40) (1 % 20)

strength :: PropertyT IO ()
strength = do
    supplied <- F.fixture F.trials
    contract <- contractFor supplied
    observed <- evalEither (U.observe supplied)
    let finiteFinding = established contract observed
        populationContract = mapLoss (\requested -> requested {U.standard = U.HoeffdingPopulation population}) contract
    first <- unknown (U.admit contract (U.establish (U.Claim (U.scope observed) U.LossIncrease 1) observed))
    assert (U.RequirementsMismatch `elem` first)
    second <- unknown (U.admit populationContract finiteFinding)
    assert (U.RequirementsMismatch `elem` second)

domain :: PropertyT IO ()
domain = do
    supplied <- F.fixture F.trials
    contract <- contractFor supplied
    observed <- evalEither (U.observe supplied)
    other <- F.fixture [trial {F.answer = "#### 99"} | trial <- F.trials]
    alteredPolicy <- evalEither (Policy.successor (replicate 64 'b') (U.candidateImplementation contract))
    forM_ [contract {U.declaredDomain = U.domain other}, contract {U.candidateImplementation = alteredPolicy}, contract {U.maximumContext = 2}] $ \changed -> do
        reasons <- unknown (U.admit changed (established contract observed))
        assert (not (null reasons))

reliance :: PropertyT IO ()
reliance = do
    supplied <- F.fixture F.trials
    contract <- contractFor supplied
    observed <- evalEither (U.observe supplied)
    forM_ [U.ParameterMeaning, U.MeasurementMeaning, U.MeasurementInput, U.ExecutionAuthenticity, U.ContractFrozen, U.AcceptanceIsolation, U.RequirementsJustified, U.SelectionControl] $ \kind -> do
        let requested = contract {U.reliance = filter ((/= kind) . U.premise) (U.reliance contract)}
        reasons <- unknown (U.admit requested (established requested observed))
        assert (not (null reasons) && all missing reasons)
  where
    missing (U.MissingReliance _) = True
    missing _ = False

invalid :: PropertyT IO ()
invalid = do
    supplied <- F.fixture F.trials
    contract <- contractFor supplied
    observed <- evalEither (U.observe supplied)
    let invalidPopulation = population {U.familyAlpha = 1 % 100}
    forM_ [contract {U.purpose = " "}, contract {U.reliance = U.reliance contract ++ U.reliance contract}, mapLoss (\requested -> requested {U.standard = U.HoeffdingPopulation invalidPopulation}) contract] $ \changed -> do
        reasons <- unknown (U.admit changed (established changed observed))
        assert (any invalidReason reasons)
    let obligation = E.Obligation "unproved-state-preservation" "fixture/v1" "none" "domain" "binding"
        derived = mapLoss (\requested -> requested {U.standard = U.ConditionalDerivation [obligation]}) contract
    reasons <- unknown (U.admit derived (established derived observed))
    reasons === [U.FindingUnknown (E.TaskLossProblem (U.UnsupportedDerivation [obligation]))]
  where
    invalidReason (U.InvalidContract _) = True
    invalidReason _ = False

statistical :: PropertyT IO ()
statistical = do
    let values = [F.Trial (show question ++ "/" ++ show seed) ("question " ++ show question) seed "#### 12" "#### 12" "#### 12" False | question <- [1 .. 32 :: Int], seed <- [1 .. 4]]
    supplied <- F.fixture values
    contract <- contractFor supplied
    observed <- evalEither (U.observe supplied)
    let declared = population {U.referenceAlpha = 1 % 20, U.regressionAlpha = 1 % 20, U.familyAlpha = 1 % 10}
        target = U.PopulationClaim (U.scope observed) declared U.LossIncrease (1 % 50)
    case U.confidence target observed of
        Nothing -> failure
        Just bound -> do
            U.unitCount bound === 32
            U.alpha bound === 1 % 20
            assert (U.width bound > 432704 % 1000000 && U.width bound < 432705 % 1000000)
    case U.finding (U.establish target observed) of
        E.Unknown (E.TaskLossProblem (U.InsufficientLossBound U.LossIncrease budget bound)) -> do
            budget === 1 % 50
            U.unitCount bound === 32
        other -> annotateShow other >> failure
    let requested = mapLoss (\value -> value {U.referenceCeiling = U.Budget (1 % 4) "Fixture bound only", U.regressionCeiling = U.Budget (1 % 2) "Fixture bound only", U.standard = U.HoeffdingPopulation population}) contract
    admitted <- accepted (U.admit requested (established requested observed))
    [metric | E.HoeffdingBound metric _ <- E.methods (U.evidence admitted)] === [U.ReferenceLoss, U.LossIncrease]
    assert (any ((== U.IndependentUnits) . U.premise . U.supporting) (U.conditions admitted))

encoding :: PropertyT IO ()
encoding = do
    supplied <- F.fixture F.trials
    contract <- contractFor supplied
    let relations =
            [N.SameTokens, N.SameBehaviorBits, N.SameTermination, N.PrefixLogRatioWithin (1 % 7), N.PathLogRatioWithin (1 % 9), N.ModelSubstitution]
                ++ [N.ScoredPathLogRatioWithin side (1 % 11) | side <- [N.Reference, N.Candidate]]
                ++ [N.FullVocabularyKLWithin side direction (1 % 13) | side <- [N.Reference, N.Candidate], direction <- [N.ReferenceToCandidate, N.CandidateToReference]]
        criterion = (U.criterion contract) {U.numericalRequirements = U.NumericalRequirement N.SameTermination [] "Fixture" : [U.NumericalRequirement relation [0, 4] "Codec fixture only" | relation <- relations], U.invarianceRequirements = [U.InvarianceRequirement relation 3 "Codec fixture only" | relation <- [N.SameBehaviorBits, N.SameTokens, N.SameTermination]]}
        obligation = E.Obligation "unproved" (Bytes.pack [0, 255]) "observation" "domain" "binding"
        standards = [U.FiniteDomain, U.HoeffdingPopulation population, U.EmpiricalBernsteinPopulation population, U.ConditionalDerivation [obligation]]
        bytes = Lazy.toStrict . encode
        decode = U.decodeContract
    forM_ standards $ \standard -> do
        let selected = mapLoss (\value -> value {U.standard = standard}) contract {U.criterion = criterion}
        decode (bytes (U.describeContract selected)) === Right selected
    let original = U.describeContract contract
        invalidDenominator = object ["limit" .= object ["numerator" .= (1 :: Int), "denominator" .= (0 :: Int)], "rationale" .= String "invalid fixture"]
        malformed =
            [ change "unexpected" Null original
            , change "format" (String "unknown") original
            , change "domain" Null original
            , change "measurement" (change "program_hex" (String "invalid hex") (field "measurement" original)) original
            , change "criterion" (change "loss" (change "reference_ceiling" invalidDenominator (field "loss" (field "criterion" original))) (field "criterion" original)) original
            ]
    forM_ malformed $ \value -> assert (isLeft (decode (bytes value)))
    assert (isLeft (decode ("{\"format\":\"invar-use-contract\"," <> Bytes.drop 1 (bytes original))))
    assert (isLeft (decode (bytes (change "format" (String "invar-use-contract-v1") original))))

withInvariance :: N.Relation -> Natural -> U.UseContract -> U.UseContract
withInvariance relation count contract = contract {U.criterion = (U.criterion contract) {U.invarianceRequirements = [U.InvarianceRequirement relation count "Fixture invariance only"]}}

alterRepeat :: Int -> String -> (Value -> Value) -> U.BoundRun -> PropertyT IO U.BoundRun
alterRepeat position stage alter supplied = case U.cases supplied of
    [] -> failure
    first : remaining -> do
        modified <- traverse (\(index, run) -> if index == position then alterStage stage alter run else pure run) (zip [0 ..] (U.repeats first))
        pure supplied {U.cases = first {U.repeats = modified} : remaining}

alterStage :: String -> (Value -> Value) -> N.Run -> PropertyT IO N.Run
alterStage stage alter run = do
    events <- evalEither (traverse eitherDecodeStrict (Lines.lines (N.logBytes run)))
    let update event
            | field "stage" event == String (Text.pack stage) = alter event
            | otherwise = event
    pure run {N.logBytes = wire (map update events)}

invariance :: PropertyT IO ()
invariance = do
    supplied <- F.repeated 2 F.trials
    contract <- withInvariance N.SameBehaviorBits 3 <$> contractFor supplied
    observed <- evalEither (U.observe supplied)
    length (U.units observed) === 2
    length (U.numerical observed) === 4
    U.mean U.LossIncrease observed === Just (1 % 6)
    admitted <- accepted (U.admit contract (established contract observed))
    case E.conclusion (U.evidence admitted) of
        E.All conjuncts -> length conjuncts === 2 + 4 + 4 * 2
        other -> annotateShow other >> failure
    length (E.assumptions (U.evidence admitted)) === 53 + 8 * 6
    length (U.conditions admitted) === 57 + 8 * 6 + 8
    length [() | U.ReliedOn _ selected <- U.conditions admitted, U.premise selected == U.ScheduleVariation] === 8
    field "unit_count" (U.describe observed) === Number 2
    field "sample_count" (U.describe observed) === Number 4

divergentRepeat :: PropertyT IO ()
divergentRepeat = do
    supplied <- F.repeated 2 F.trials
    contract <- withInvariance N.SameBehaviorBits 3 <$> contractFor supplied
    changed <- alterRepeat 1 "result" (change "behavior" (toJSON [-0.75, -0.25 :: Double]) . change "behavior_bits" (toJSON [0xbf400000, 0xbe800000 :: Word32])) supplied
    observed <- evalEither (U.observe changed)
    let tokens = withInvariance N.SameTokens 3 contract
    _ <- accepted (U.admit tokens (established tokens observed))
    case U.admit contract (established contract observed) of
        U.Rejected counterexample -> case E.witness counterexample of
            E.NumericalWitness measured -> do
                field "tokens_equal" (N.describe measured) === Bool True
                field "behavior_bits_equal" (N.describe measured) === Bool False
                assert (N.scopeId (N.scope measured) `notElem` map (N.scopeId . N.scope . snd) (NonEmpty.toList (U.numerical observed)))
            other -> annotateShow other >> failure
        other -> annotateShow other >> failure

invarianceContract :: PropertyT IO ()
invarianceContract = do
    supplied <- F.repeated 1 F.trials
    contract <- contractFor supplied
    observed <- evalEither (U.observe supplied)
    let three = withInvariance N.SameBehaviorBits 3 contract
    reasons <- unknown (U.admit three (established three observed))
    reasons === [U.InsufficientExecutions (U.Key 0 name) 3 2 | name <- ["a1", "a2", "a3", "b1"]]
    let two = withInvariance N.SameBehaviorBits 2 contract
    _ <- accepted (U.admit two (established two observed))
    forM_ [U.InvarianceRequirement N.SameBehaviorBits 1 "Fixture", U.InvarianceRequirement (N.PrefixLogRatioWithin 1) 2 "Fixture", U.InvarianceRequirement N.SameTokens 2 " "] $ \requirement -> do
        let changed = contract {U.criterion = (U.criterion contract) {U.invarianceRequirements = [requirement]}}
        problems <- unknown (U.admit changed (established changed observed))
        assert (any invalidReason problems)
    let uncovered = two {U.reliance = filter ((/= U.ScheduleVariation) . U.premise) (U.reliance contract)}
    missing <- unknown (U.admit uncovered (established uncovered observed))
    assert (not (null missing) && all unsupported missing)
    differing <- alterRepeat 0 "loaded_adapter" (change "revision" (String "other-model-revision")) supplied
    case U.observe differing of
        Left (U.MixedImplementation N.Candidate _) -> success
        other -> annotateShow other >> failure
    case U.cases supplied of
        [] -> failure
        first : remaining -> case U.observe supplied {U.cases = first {U.repeats = [N.candidate (U.paired first)]} : remaining} of
            Left (U.ReusedExecution N.Candidate _) -> success
            other -> annotateShow other >> failure
  where
    invalidReason (U.InvalidContract _) = True
    invalidReason _ = False
    unsupported (U.MissingReliance _) = True
    unsupported _ = False

mapLoss :: (U.LossRequirement -> U.LossRequirement) -> U.UseContract -> U.UseContract
mapLoss transform contract = contract {U.criterion = selected {U.lossRequirement = transform <$> U.lossRequirement selected}}
  where
    selected = U.criterion contract

transfer :: PropertyT IO ()
transfer = do
    supplied <- F.fixture F.trials
    contract <- contractFor supplied
    observed <- evalEither (U.observe supplied)
    let recorded = U.candidateImplementation contract
        described (adapter, assembly) = Policy.describe (Policy.model recorded, Policy.revision recorded) (adapter, Policy.tokenizer recorded, Policy.base recorded, assembly)
    current <- evalEither (described (Policy.adapter recorded, replicate 64 '1'))
    other <- evalEither (described (Policy.adapter recorded, replicate 64 '2'))
    reweighted <- evalEither (described (replicate 64 'b', replicate 64 '1'))
    assert (Policy.assembly recorded `notElem` [Policy.assembly current, Policy.assembly other])
    let moved = contract {U.referenceImplementation = current, U.candidateImplementation = current}
        carried = moved {U.transfers = [U.Transfer side recorded "Reviewed diff changes no numerical dependency" | side <- [N.Reference, N.Candidate]]}
        admitting requested = U.admit requested (established requested observed)
    mismatched <- unknown (admitting moved)
    assert (not (null mismatched) && all isMismatch mismatched)
    admitted <- accepted (admitting carried)
    let relied = [U.premise (U.supporting condition) | condition <- U.conditions admitted]
    length (filter (== U.ImplementationPreservation) relied) === 2
    length (U.conditions admitted) === 59
    U.admissionScope admitted === U.scope observed
    uncovered <- unknown (admitting carried {U.reliance = filter ((/= U.ImplementationPreservation) . U.premise) (U.reliance carried)})
    assert (not (null uncovered) && all isMissing uncovered)
    forM_
        [ carried {U.transfers = [U.Transfer N.Candidate current "Identical"]}
        , carried {U.transfers = [U.Transfer N.Candidate recorded " "]}
        , carried {U.referenceImplementation = reweighted, U.candidateImplementation = reweighted}
        , carried {U.transfers = U.transfers carried ++ U.transfers carried}
        ]
        $ \changed -> do
            reasons <- unknown (admitting changed)
            assert (any isInvalid reasons)
    unrelated <- unknown (admitting moved {U.transfers = [U.Transfer side other "Reviewed diff" | side <- [N.Reference, N.Candidate]]})
    assert (any isMismatch unrelated)
    oneSided <- unknown (admitting moved {U.transfers = [U.Transfer N.Candidate recorded "Reviewed diff"]})
    assert (U.ImplementationMismatch N.Reference (fst (NonEmpty.head (U.numerical observed))) `elem` oneSided)
    other' <- F.fixture [trial {F.answer = "#### 99"} | trial <- F.trials]
    widened <- unknown (U.admit carried {U.declaredDomain = U.domain other'} (established carried observed))
    assert (U.DomainMismatch `elem` widened)
    U.decodeContract (Lazy.toStrict (encode (U.describeContract carried))) === Right carried
    let original = U.describeContract contract
        previous = change "format" (String "invar-use-contract") (Object (KeyMap.delete "transfers" (object' original)))
    U.decodeContract (Lazy.toStrict (encode previous)) === Right contract
    assert (isLeft (U.decodeContract (Lazy.toStrict (encode (change "format" (String "invar-use-contract") original)))))
  where
    isMismatch (U.ImplementationMismatch _ _) = True
    isMismatch _ = False
    isMissing (U.MissingReliance _) = True
    isMissing _ = False
    isInvalid (U.InvalidContract _) = True
    isInvalid _ = False
    object' (Object fields) = fields
    object' _ = KeyMap.empty
