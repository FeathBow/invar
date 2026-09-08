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
    Counterexample,
    refuted,
    witness,
    check,
) where

import Data.ByteString (ByteString)
import Data.List (find, nub, (\\))
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Invar.Spec.Invocation qualified as I
import Numeric.Natural (Natural)

newtype EvidenceId = EvidenceId Natural
    deriving (Eq, Ord, Show)

data Obligation = Obligation
    { predicate :: String
    , specification :: ByteString
    , observation :: String
    , domain :: ByteString
    , binding :: ByteString
    }
    deriving (Eq, Show)

data Claim = External Obligation | OutputEqual I.Completion ByteString | All [Claim]
    deriving (Eq, Show)

data Rule = Assume | Compare | Conjoin [EvidenceId] | Discharge EvidenceId [EvidenceId]
    deriving (Eq, Show)

data Node = Node {claim :: Claim, rule :: Rule}
    deriving (Eq, Show)

type Graph = Map EvidenceId Node

data Method = Hypothesis | ReportComparison | Conjunction | AssumptionDischarge
    deriving (Eq, Show)

data Problem
    = MissingEvidence EvidenceId
    | Cycle EvidenceId
    | WrongConclusion Claim Claim
    | UnsupportedComparison Claim
    | UnmatchedDischarge Claim
    deriving (Eq, Show)

data Certificate = Certificate Claim [Claim] [Method]
    deriving (Eq, Show)

data Counterexample = Counterexample Claim I.Completion ByteString
    deriving (Eq, Show)

data Verdict = Accept Certificate | Refute Counterexample | Unknown Problem
    deriving (Eq, Show)

conclusion :: Certificate -> Claim
conclusion (Certificate result _ _) = result

assumptions :: Certificate -> [Claim]
assumptions (Certificate _ remaining _) = remaining

methods :: Certificate -> [Method]
methods (Certificate _ _ used) = used

refuted :: Counterexample -> Claim
refuted (Counterexample target _ _) = target

witness :: Counterexample -> (I.Completion, ByteString)
witness (Counterexample _ actual expected) = (actual, expected)

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
references _ = []

validateConclusion :: Graph -> Node -> Either Problem ()
validateConclusion graph node = case rule node of
    Conjoin names -> traverse target names >>= matches . All
    Discharge premise _ -> target premise >>= matches
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
        Conjoin _ -> conjunction (claim node) children
        Discharge _ _ -> discharge (claim node) children

firstProblem :: [Verdict] -> Maybe Problem
firstProblem [] = Nothing
firstProblem (Unknown problem : _) = Just problem
firstProblem (_ : rest) = firstProblem rest

compareOutput :: Claim -> Verdict
compareOutput target@(OutputEqual actual expected)
    | I.completedOutput actual == expected = Accept (Certificate target [] [ReportComparison])
    | otherwise = Refute (Counterexample target actual expected)
compareOutput target = Unknown (UnsupportedComparison target)

conjunction :: Claim -> [Verdict] -> Verdict
conjunction target children = case firstRefutation children of
    Just (Counterexample _ actual expected) -> Refute (Counterexample target actual expected)
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
