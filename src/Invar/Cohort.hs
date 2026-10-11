{-# LANGUAGE RoleAnnotations #-}
{-# LANGUAGE TypeFamilies #-}

module Invar.Cohort (
    TaskOf (..),
    Task,
    Graded (..),
    Ungraded (..),
    Grade (..),
    DefinitionOf (..),
    Definition,
    CohortOf,
    Cohort,
    MemberOf,
    Member,
    ObservationOf,
    Observation,
    BatchOf,
    Batch,
    Error (..),
    withCohort,
    validateMembers,
    group,
    rule,
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
import Invar.Infer.Trajectory (Trajectory)
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Reward qualified as Reward
import Numeric.Natural (Natural)

data TaskOf grade = Task {name :: String, plan :: I.Plan, grade :: grade}

type Task = TaskOf Graded

data Graded = Graded String Reward.Rule
    deriving (Eq, Show)

data Ungraded = Ungraded
    deriving (Eq, Show)

class Grade grade where
    type Assessment grade
    validateGrades :: [grade] -> Either Error ()
    assess :: grade -> Trajectory -> Either Error (Assessment grade)

instance Grade Graded where
    type Assessment Graded = Reward.Scored
    validateGrades graded = do
        let groups = Map.fromListWith (+) [(membership, 1 :: Natural) | Graded membership _ <- graded]
        when (Map.member "" groups) (Left InvalidIdentity)
        when (any (< minimumGroup) groups) (Left SingletonGroup)
      where
        minimumGroup = 2
    assess (Graded _ chosen) output = either (Left . RewardError) Right (Reward.score chosen (Trajectory.text output) (Trajectory.truncated output))

instance Grade Ungraded where
    type Assessment Ungraded = ()
    validateGrades _ = Right ()
    assess _ _ = Right ()

data DefinitionOf grade = Definition {policy :: String, tasks :: [TaskOf grade]}

type Definition = DefinitionOf Graded

type role CohortOf nominal nominal
newtype CohortOf scope grade = Cohort [MemberOf scope grade]

type Cohort scope = CohortOf scope Graded

type role MemberOf nominal nominal
data MemberOf scope grade = Member Natural (TaskOf grade)

type Member scope = MemberOf scope Graded

type role ObservationOf nominal nominal
data ObservationOf scope grade = Observation (MemberOf scope grade) Trajectory (Assessment grade)

type Observation scope = ObservationOf scope Graded

type role BatchOf nominal nominal
newtype BatchOf scope grade = Batch [ObservationOf scope grade]

type Batch scope = BatchOf scope Graded

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

withCohort :: (Grade grade) => DefinitionOf grade -> (forall scope. CohortOf scope grade -> result) -> Either Error result
withCohort definition continuation = do
    validate definition
    pure (continuation (Cohort (zipWith Member [0 ..] (tasks definition))))

validate :: (Grade grade) => DefinitionOf grade -> Either Error ()
validate definition = do
    validateMembers [(name task, grade task) | task <- tasks definition]
    mapM_ samePolicy (tasks definition)
  where
    samePolicy task = unless (I.artifact (I.requested (plan task)) == policy definition) (Left (PolicyMismatch (name task)))

validateMembers :: (Grade grade) => [(String, grade)] -> Either Error ()
validateMembers entries = do
    let names = map fst entries
    when (null entries) (Left EmptyCohort)
    when (any null names) (Left InvalidIdentity)
    unless (Set.size (Set.fromList names) == length names) (Left DuplicateMember)
    validateGrades (map snd entries)

group :: TaskOf Graded -> String
group task = let Graded membership _ = grade task in membership

rule :: TaskOf Graded -> Reward.Rule
rule task = let Graded _ chosen = grade task in chosen

members :: CohortOf scope grade -> [MemberOf scope grade]
members (Cohort entries) = entries

planned :: MemberOf scope grade -> I.Plan
planned (Member _ task) = plan task

record :: (Grade grade) => MemberOf scope grade -> Trajectory -> Either Error (ObservationOf scope grade)
record member@(Member _ task) output = do
    unless (Trajectory.request output == I.requested (plan task)) (Left (RequestMismatch (name task)))
    evaluated <- assess (grade task) output
    pure (Observation member output evaluated)

admit :: CohortOf scope grade -> [ObservationOf scope grade] -> Either Error (BatchOf scope grade)
admit (Cohort entries) delivered = do
    indexed <- foldM insert Map.empty delivered
    unless (Map.size indexed == length entries) (Left IncompleteCohort)
    ordered <- maybe (Left IncompleteCohort) Right (traverse (\(Member index _) -> Map.lookup index indexed) entries)
    pure (Batch ordered)
  where
    insert accumulated observation@(Observation (Member index _) _ _) = do
        when (Map.member index accumulated) (Left RepeatedResult)
        pure (Map.insert index observation accumulated)

observations :: BatchOf scope grade -> [ObservationOf scope grade]
observations (Batch entries) = entries

observed :: ObservationOf scope grade -> Trajectory
observed (Observation _ output _) = output

source :: ObservationOf scope grade -> TaskOf grade
source (Observation (Member _ task) _ _) = task

reward :: ObservationOf scope Graded -> Rational
reward = Reward.value . scored

scored :: ObservationOf scope grade -> Assessment grade
scored (Observation _ _ evaluated) = evaluated
