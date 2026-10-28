{-# LANGUAGE Safe #-}

module Invar.Spec.Evidence (
    EvidenceId (..),
    Obligation (..),
    Claim (..),
    Rule (..),
    Node (..),
    Graph,
    Method (..),
    Problem (..),
    Verdict (..),
    Certificate,
    conclusion,
    assumptions,
    methods,
    methodName,
    methodNames,
    bounds,
    Counterexample,
    Witness (..),
    refuted,
    witness,
    check,
) where

import Crypto.Hash.SHA256 qualified as SHA256
import Data.ByteString (ByteString)
import Data.List (find, nub, (\\))
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text qualified as Text
import Data.Text.Encoding qualified as Text
import Invar.Paths qualified as Paths
import Invar.Spec.Invocation qualified as I
import Invar.Spec.Numerical qualified as N
import Invar.Spec.Obligation (Obligation (..))
import Invar.Spec.Use qualified as U
import Invar.Spec.UseContract qualified as Contract
import Numeric.Natural (Natural)

newtype EvidenceId = EvidenceId Natural
    deriving (Eq, Ord, Show)

data Claim = External Obligation | OutputEqual I.Completion ByteString | Numerical N.Claim | Paths Paths.ScopedClaim | TaskLoss U.Claim | All [Claim] | Implies Claim Claim
    deriving (Eq, Show)

data Rule = Assume | Compare | Observe N.Observed | ObservePaths Paths.Observed | ObserveTaskLoss U.Observed | Conjoin [EvidenceId] | Discharge EvidenceId [EvidenceId] | Apply EvidenceId EvidenceId
    deriving (Eq, Show)

data Node = Node {claim :: Claim, rule :: Rule}
    deriving (Eq, Show)

type Graph = Map EvidenceId Node

data Method = Hypothesis | ReportComparison | NumericalObservation | PathObservation | TaskLossObservation | HoeffdingBound U.Metric U.Confidence | EmpiricalBernsteinBound U.Metric U.Confidence | Conjunction | AssumptionDischarge | ImplicationElimination
    deriving (Eq, Show)

data Problem
    = MissingEvidence EvidenceId
    | Cycle EvidenceId
    | WrongConclusion Claim Claim
    | UnsupportedComparison Claim
    | UnmatchedDischarge Claim
    | UnsupportedApplication Claim
    | RefutedPremise Claim
    | UnsupportedObservation Claim
    | NumericalProblem N.Problem
    | PathProblem Paths.Problem
    | TaskLossProblem U.Problem
    deriving (Eq, Show)

data Certificate = Certificate Claim [Claim] [Method]
    deriving (Eq, Show)

data Witness = OutputWitness I.Completion ByteString | NumericalWitness N.Observed | PathWitness Paths.Observed | TaskLossWitness U.Observed
    deriving (Eq, Show)

data Counterexample = Counterexample Claim Witness
    deriving (Eq, Show)

data Verdict = Accept Certificate | Refute Counterexample | Unknown Problem
    deriving (Eq, Show)

conclusion :: Certificate -> Claim
conclusion (Certificate result _ _) = result

assumptions :: Certificate -> [Claim]
assumptions (Certificate _ remaining _) = remaining

methods :: Certificate -> [Method]
methods (Certificate _ _ used) = used

methodName :: Method -> String
methodName method = case method of
    HoeffdingBound {} -> "HoeffdingBound"
    EmpiricalBernsteinBound {} -> "EmpiricalBernsteinBound"
    other -> show other

methodNames :: Certificate -> [String]
methodNames = nub . map methodName . methods

bounds :: Verdict -> [(U.Metric, U.Confidence)]
bounds verdict = case verdict of
    Accept certificate -> [bound | method <- methods certificate, Just bound <- [used method]]
    Unknown (TaskLossProblem (U.InsufficientLossBound metric _ value)) -> [(metric, value)]
    _ -> []
  where
    used (HoeffdingBound metric value) = Just (metric, value)
    used (EmpiricalBernsteinBound metric value) = Just (metric, value)
    used _ = Nothing

refuted :: Counterexample -> Claim
refuted (Counterexample target _) = target

witness :: Counterexample -> Witness
witness (Counterexample _ actual) = actual

check :: Graph -> EvidenceId -> Verdict
check graph root = fst (visit Set.empty Map.empty root)
  where
    visit path cache name
        | Just known <- Map.lookup name cache = (known, cache)
        | Set.member name path = (Unknown (Cycle name), cache)
        | otherwise = case Map.lookup name graph of
            Nothing -> (Unknown (MissingEvidence name), cache)
            Just node ->
                let (children, updated) = descend (Set.insert name path) cache (references (rule node))
                    result = either Unknown (evaluate node children) (validateConclusion graph node)
                 in (result, Map.insert name result updated)
    descend _ cache [] = ([], cache)
    descend path cache (name : rest) =
        let (first, updated) = visit path cache name
            (remaining, finished) = descend path updated rest
         in (first : remaining, finished)

references :: Rule -> [EvidenceId]
references (Conjoin names) = names
references (Discharge premise names) = premise : names
references (Apply implication premise) = [implication, premise]
references _ = []

validateConclusion :: Graph -> Node -> Either Problem ()
validateConclusion graph node = case rule node of
    Conjoin names -> traverse target names >>= matches . All
    Discharge premise _ -> target premise >>= matches
    Apply implication premise -> do
        declared <- target implication
        case declared of
            Implies antecedent consequent -> do
                actual <- target premise
                if actual == antecedent then matches consequent else Left (WrongConclusion antecedent actual)
            other -> Left (UnsupportedApplication other)
    _ -> Right ()
  where
    target name = maybe (Left (MissingEvidence name)) (Right . claim) (Map.lookup name graph)
    matches expected
        | claim node == expected = Right ()
        | otherwise = Left (WrongConclusion expected (claim node))

evaluate :: Node -> [Verdict] -> () -> Verdict
evaluate node children () = case firstProblem children of
    Just problem -> Unknown problem
    Nothing -> case rule node of
        Assume -> Accept (Certificate (claim node) [claim node] [Hypothesis])
        Compare -> compareOutput (claim node)
        Observe observed -> compareNumerical (claim node) observed
        ObservePaths observed -> comparePaths (claim node) observed
        ObserveTaskLoss observed -> compareTaskLoss (claim node) observed
        Conjoin _ -> conjunction (claim node) children
        Discharge _ _ -> discharge (claim node) children
        Apply _ _ -> application (claim node) children

firstProblem :: [Verdict] -> Maybe Problem
firstProblem [] = Nothing
firstProblem (Unknown problem : _) = Just problem
firstProblem (_ : rest) = firstProblem rest

compareOutput :: Claim -> Verdict
compareOutput target@(OutputEqual actual expected)
    | I.completedOutput actual == expected = Accept (Certificate target [] [ReportComparison])
    | otherwise = Refute (Counterexample target (OutputWitness actual expected))
compareOutput target = Unknown (UnsupportedComparison target)

compareNumerical :: Claim -> N.Observed -> Verdict
compareNumerical target@(Numerical expected) observed = case N.judge expected observed of
    N.Satisfied -> Accept (Certificate target obligations [NumericalObservation])
    N.Violated -> Refute (Counterexample target (NumericalWitness observed))
    N.Insufficient problem -> Unknown (NumericalProblem problem)
  where
    obligations = map External (N.premises observed)
compareNumerical target _ = Unknown (UnsupportedObservation target)

comparePaths :: Claim -> Paths.Observed -> Verdict
comparePaths target@(Paths expected) observed = case Paths.judgeScoped expected observed of
    Paths.Satisfied _ -> Accept (Certificate target obligations [PathObservation])
    Paths.Violated _ -> Refute (Counterexample target (PathWitness observed))
    Paths.Insufficient problem -> Unknown (PathProblem problem)
  where
    obligations = map External (pathPremises expected observed)
comparePaths target _ = Unknown (UnsupportedObservation target)

pathPremises :: Paths.ScopedClaim -> Paths.Observed -> [Obligation]
pathPremises claim observed = observations ++ correspondence
  where
    observations =
        [ obligation kind ("/" ++ label) (digest (Paths.describeObservation value))
        | (label, value) <- [("learner", Paths.learner observed), ("engine", Paths.engine observed)]
        , kind <- [Contract.PathAuthenticity, Contract.PathBehavior]
        ]
    correspondence = [obligation Contract.ForwardCorrespondence "" (digest (Paths.describeCorrespondence checked)) | Just checked <- [Paths.observedCorrespondence observed]]
    Paths.ScopeId domain = Paths.scopeOf claim
    obligation kind suffix binding =
        let (name, specification, observation) = Contract.premiseDescription kind
         in Obligation (name ++ suffix) specification observation domain binding
    digest value = SHA256.hash (Text.encodeUtf8 (Text.pack (show value)))

compareTaskLoss :: Claim -> U.Observed -> Verdict
compareTaskLoss target@(TaskLoss expected) observed = case U.judge expected observed of
    U.Satisfied bound -> Accept (Certificate target (map External (U.claimPremises expected observed)) (TaskLossObservation : used bound))
    U.Violated -> Refute (Counterexample target (TaskLossWitness observed))
    U.Insufficient problem -> Unknown (TaskLossProblem problem)
  where
    used bound = case (expected, bound) of
        (U.PopulationClaim _ _ metric _, Just value) -> [HoeffdingBound metric value]
        (U.EmpiricalBernsteinClaim _ _ metric _, Just value) -> [EmpiricalBernsteinBound metric value]
        _ -> []
compareTaskLoss target _ = Unknown (UnsupportedObservation target)

conjunction :: Claim -> [Verdict] -> Verdict
conjunction target children = case firstRefutation children of
    Just (Counterexample _ actual) -> Refute (Counterexample target actual)
    Nothing -> case certificates children of
        Left problem -> Unknown problem
        Right accepted ->
            Accept (Certificate target (nub (concatMap assumptions accepted)) (nub (Conjunction : concatMap methods accepted)))

firstRefutation :: [Verdict] -> Maybe Counterexample
firstRefutation [] = Nothing
firstRefutation (Refute counterexample : _) = Just counterexample
firstRefutation (_ : rest) = firstRefutation rest

certificates :: [Verdict] -> Either Problem [Certificate]
certificates = traverse accepted
  where
    accepted (Accept certificate) = Right certificate
    accepted (Unknown problem) = Left problem
    accepted (Refute counterexample) = Left (UnmatchedDischarge (refuted counterexample))

application :: Claim -> [Verdict] -> Verdict
application target children = case firstRefutation children of
    Just counterexample -> Unknown (RefutedPremise (refuted counterexample))
    Nothing -> case certificates children of
        Left problem -> Unknown problem
        Right accepted -> Accept (Certificate target (nub (concatMap assumptions accepted)) (nub (ImplicationElimination : concatMap methods accepted)))

discharge :: Claim -> [Verdict] -> Verdict
discharge _ (Refute counterexample : _) = Refute counterexample
discharge target children = case certificates children of
    Left problem -> Unknown problem
    Right [] -> Unknown (UnmatchedDischarge target)
    Right (premise : proofs) ->
        let original = assumptions premise
            provided = map conclusion proofs
            remaining = nub ((original \\ provided) ++ concatMap assumptions proofs)
            used = nub (AssumptionDischarge : concatMap methods (premise : proofs))
         in case find (`notElem` original) provided of
                Just extra -> Unknown (UnmatchedDischarge extra)
                Nothing -> Accept (Certificate target remaining used)
