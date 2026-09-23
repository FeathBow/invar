{-# LANGUAGE Safe #-}

module Invar.Spec.Numerical (
    Side (..),
    Source (..),
    ScopeId (..),
    Scope (..),
    Path (..),
    Observed (..),
    scope,
    path,
    scopeId,
    scores,
    distributions,
    Distribution (..),
    Direction (..),
    Relation (..),
    Claim (..),
    Problem (..),
    Judgement (..),
    judge,
    premises,
) where

import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Invar.Numerical.Distribution qualified as KL
import Invar.Spec.Obligation (Obligation)
import Invar.Spec.Obligation qualified as Obligation
import Invar.Spec.Score (Source (..))
import Invar.Spec.Score qualified as Score
import Numeric.Natural (Natural)

data Side = Reference | Candidate
    deriving (Eq, Show)

newtype ScopeId = ScopeId ByteString
    deriving (Eq, Show)

data Scope = Scope ScopeId Source Source [Natural] [(Side, Score.Fact)] [(Side, Side, Score.Fact)]
    deriving (Eq, Show)

data Path = Path
    { firstDivergence :: Maybe Natural
    , matchingSteps :: Natural
    , prefixLogRatio :: Rational
    , referenceLength :: Natural
    , candidateLength :: Natural
    , referenceTruncated :: Bool
    , candidateTruncated :: Bool
    , tokensEqual :: Bool
    , behaviorBitsEqual :: Bool
    }
    deriving (Eq, Show)

data Distribution = Distribution {step :: Natural, forward :: KL.Bounds, backward :: KL.Bounds}
    deriving (Eq, Show)

data Observed = Observed Scope Path [(Side, [Distribution])]
    deriving (Eq, Show)

scope :: Observed -> Scope
scope (Observed value _ _) = value

path :: Observed -> Path
path (Observed _ value _) = value

scopeId :: Scope -> ScopeId
scopeId (Scope identity _ _ _ _ _) = identity

scores :: Observed -> [(Side, Score.Fact)]
scores observed = let Scope _ _ _ _ values _ = scope observed in values

distributions :: Observed -> [(Side, [Distribution])]
distributions (Observed _ _ values) = values

data Direction = ReferenceToCandidate | CandidateToReference
    deriving (Eq, Show)

data Relation
    = SameTokens
    | SameBehaviorBits
    | SameTermination
    | PrefixLogRatioWithin Rational
    | PathLogRatioWithin Rational
    | ScoredPathLogRatioWithin Side Rational
    | FullVocabularyKLWithin Side Direction Rational
    | ModelSubstitution
    deriving (Eq, Show)

data Claim = Claim Scope Relation
    deriving (Eq, Show)

data Problem
    = ScopeMismatch ScopeId ScopeId
    | InvalidBudget Rational
    | MissingCrossScoring
    | MissingScoredPath Side
    | MissingFullVocabulary
    | KLReductionUncertain Side Direction [Natural]
    | UnsupportedModelDerivation
    deriving (Eq, Show)

data Judgement = Satisfied | Violated | Insufficient Problem
    deriving (Eq, Show)

judge :: Claim -> Observed -> Judgement
judge (Claim expected relation) observed
    | expected /= scope observed = Insufficient (ScopeMismatch (scopeId expected) (scopeId (scope observed)))
    | otherwise = judgeRelation relation observed

judgeRelation :: Relation -> Observed -> Judgement
judgeRelation relation observed = case relation of
    SameTokens -> decide (tokensEqual measured)
    SameBehaviorBits -> decide (behaviorBitsEqual measured)
    SameTermination -> decide (sameTermination measured)
    PrefixLogRatioWithin budget -> budgeted budget (bounded budget (Score.Finite (prefixLogRatio measured)))
    PathLogRatioWithin budget -> budgeted budget (commonPath budget measured)
    ScoredPathLogRatioWithin side budget ->
        budgeted budget (maybe (Insufficient (MissingScoredPath side)) (bounded budget . Score.logRatio) (lookup side (scores observed)))
    FullVocabularyKLWithin side direction budget -> budgeted budget (distributionBound (side, direction) budget observed)
    ModelSubstitution -> Insufficient UnsupportedModelDerivation
  where
    measured = path observed

sameTermination :: Path -> Bool
sameTermination measured = referenceLength measured == candidateLength measured && referenceTruncated measured == candidateTruncated measured

commonPath :: Rational -> Path -> Judgement
commonPath budget measured
    | tokensEqual measured && sameTermination measured = bounded budget (Score.Finite (prefixLogRatio measured))
    | otherwise = Insufficient MissingCrossScoring

budgeted :: Rational -> Judgement -> Judgement
budgeted budget result
    | budget < 0 = Insufficient (InvalidBudget budget)
    | otherwise = result

bounded :: Rational -> Score.LogRatio -> Judgement
bounded budget (Score.Finite value) = decide (abs value <= budget)
bounded _ Score.PositiveInfinity = Violated

decide :: Bool -> Judgement
decide True = Satisfied
decide False = Violated

distributionBound :: (Side, Direction) -> Rational -> Observed -> Judgement
distributionBound (side, direction) budget observed = case lookup side (distributions observed) of
    Nothing -> Insufficient MissingFullVocabulary
    Just values
        | any (violates . snd) selected -> Violated
        | null uncertain -> Satisfied
        | otherwise -> Insufficient (KLReductionUncertain side direction uncertain)
      where
        selected = [(step value, case direction of ReferenceToCandidate -> forward value; CandidateToReference -> backward value) | value <- values]
        uncertain = [position | (position, bounds) <- selected, not (within bounds)]
  where
    violates KL.InfiniteKL = True
    violates (KL.FiniteBounds lower _) = lower > budget
    within KL.InfiniteKL = False
    within (KL.FiniteBounds _ upper) = upper <= budget

premises :: Observed -> [Obligation]
premises observed = generation ++ scoring ++ probing
  where
    Scope (ScopeId domain) left right _ measured probes = scope observed
    generation =
        [ obligation ("finite-paired-inference/v1", "selected-token-behavior/v1") (name ++ "/" ++ side) (Bytes.pack (sourceLog source ++ ":" ++ show (sourceBinding source)))
        | (side, source) <- [("reference", left), ("candidate", right)]
        , name <- ["execution-report-authenticity", "selected-behavior-measurement", "own-cache-execution"]
        ]
    scoring =
        [ obligation ("cached-path-score/v1", "selected-path-behavior/v1") (name ++ "/" ++ show side ++ "-path") (Bytes.pack (sourceLog target ++ ":" ++ show (sourceBinding target) ++ ":source=" ++ sourceLog (Score.source score) ++ ":target-generation=" ++ sourceLog other))
        | (side, score) <- measured
        , let target = Score.target score
              other = case side of Reference -> right; Candidate -> left
        , name <- ["score-execution-report-authenticity", "scored-behavior-measurement", "own-cache-scoring", "scoring-generation-correspondence"]
        ]
    probing =
        [ obligation ("cached-distribution-probe/v1", "full-vocabulary-behavior/v1") (name ++ "/" ++ show side ++ "-path/" ++ show implementation) (Bytes.pack (sourceLog target ++ ":" ++ show (sourceBinding target) ++ ":source=" ++ sourceLog (Score.source probe) ++ ":target-generation=" ++ sourceLog other))
        | (side, implementation, probe) <- probes
        , let target = Score.target probe
              other = case implementation of Reference -> left; Candidate -> right
        , name <- ["probe-execution-report-authenticity", "full-vocabulary-behavior-measurement", "own-cache-probing", "probing-generation-correspondence", "vocabulary-coordinate-correspondence"]
        ]
    obligation (specification, observation) name = Obligation.Obligation name (Bytes.pack specification) observation domain
