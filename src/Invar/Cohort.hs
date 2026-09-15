{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RoleAnnotations #-}

module Invar.Cohort (
    Task (..),
    Definition (..),
    Cohort,
    Member,
    Observation,
    Batch,
    Error (..),
    withCohort,
    validateMembers,
    members,
    planned,
    record,
    admit,
    observations,
    observed,
    source,
    reward,
    scored,
) where

import Control.Monad (foldM, unless, when)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Invar.Infer qualified as I
import Invar.Infer.Result qualified as R
import Invar.Reward qualified as Reward
import Numeric.Natural (Natural)

data Task = Task {name :: String, group :: String, plan :: I.Plan, rule :: Reward.Rule}

data Definition = Definition {policy :: String, tasks :: [Task]}

type role Cohort nominal
newtype Cohort scope = Cohort [Member scope]

type role Member nominal
data Member scope = Member Natural Task

type role Observation nominal
data Observation scope = Observation (Member scope) R.Result Reward.Scored

type role Batch nominal
newtype Batch scope = Batch [Observation scope]

data Error
    = EmptyCohort
    | InvalidIdentity
    | DuplicateMember
    | SingletonGroup
    | PolicyMismatch String
    | RequestMismatch String
    | RepeatedResult
    | IncompleteCohort
    | RewardError Reward.Error
    deriving (Eq, Show)

withCohort :: Definition -> (forall scope. Cohort scope -> result) -> Either Error result
withCohort definition continuation = do
    validate definition
    pure (continuation (Cohort (zipWith Member [0 ..] (tasks definition))))

validate :: Definition -> Either Error ()
validate definition = do
    validateMembers [(name task, group task) | task <- tasks definition]
    mapM_ samePolicy (tasks definition)
  where
    samePolicy task = unless (I.artifact (I.requested (plan task)) == policy definition) (Left (PolicyMismatch (name task)))

validateMembers :: [(String, String)] -> Either Error ()
validateMembers entries = do
    let names = map fst entries
        groups = Map.fromListWith (+) [(membership, 1 :: Natural) | (_, membership) <- entries]
    when (null entries) (Left EmptyCohort)
    when (any null names || any (null . snd) entries) (Left InvalidIdentity)
    unless (Set.size (Set.fromList names) == length names) (Left DuplicateMember)
    when (any (< minimumGroup) groups) (Left SingletonGroup)
  where
    minimumGroup = 2

members :: Cohort scope -> [Member scope]
members (Cohort entries) = entries

planned :: Member scope -> I.Plan
planned (Member _ task) = plan task

record :: Member scope -> R.Result -> Either Error (Observation scope)
record member@(Member _ task) output = do
    unless (R.consumed output == I.requested (plan task)) (Left (RequestMismatch (name task)))
    evaluated <- either (Left . RewardError) Right (Reward.score (rule task) (R.response output) (R.truncated output))
    pure (Observation member output evaluated)

admit :: Cohort scope -> [Observation scope] -> Either Error (Batch scope)
admit (Cohort entries) delivered = do
    indexed <- foldM insert Map.empty delivered
    unless (Map.size indexed == length entries) (Left IncompleteCohort)
    ordered <- maybe (Left IncompleteCohort) Right (traverse (\(Member index _) -> Map.lookup index indexed) entries)
    pure (Batch ordered)
  where
    insert accumulated observation@(Observation (Member index _) _ _) = do
        when (Map.member index accumulated) (Left RepeatedResult)
        pure (Map.insert index observation accumulated)

observations :: Batch scope -> [Observation scope]
observations (Batch entries) = entries

observed :: Observation scope -> R.Result
observed (Observation _ output _) = output

source :: Observation scope -> Task
source (Observation (Member _ task) _ _) = task

reward :: Observation scope -> Rational
reward = Reward.value . scored

scored :: Observation scope -> Reward.Scored
scored (Observation _ _ evaluated) = evaluated
