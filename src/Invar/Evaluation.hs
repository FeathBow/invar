{-# LANGUAGE OverloadedStrings #-}

module Invar.Evaluation (
    Run (..),
    Model (..),
    Report,
    Sample,
    admit,
    describe,
    cohortValue,
    inputDigest,
    logDigest,
    policy,
    model,
    samples,
    residence,
    sampleCohort,
    sampleName,
    sampleGroup,
    sampleSeed,
    sampleReward,
    sampleTokens,
    sampleTruncated,
) where

import Control.Monad (foldM, unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Object, Value (..), object, toJSON, withObject, (.:), (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Pair, Parser, parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Invar.Artifact qualified as Artifact
import Invar.Cohort qualified as Cohort
import Invar.Infer qualified as Infer
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Model (Model (..))
import Invar.Infer.Model qualified as Model
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Session qualified as Session
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Policy qualified as Policy
import Invar.Resident qualified as Boundary
import Invar.Resident.Group qualified as Group
import Invar.Resident.Owner qualified as Owner
import Invar.Rollout qualified as Rollout
import Invar.Spec.Invocation qualified as Invocation
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)

data Run = Run {expectedPolicy :: String, exitCode :: Int, workerMode :: Rollout.Mode, description :: Maybe Policy.Description}
    deriving (Eq, Show)

data Report = Report String String Run Model [Sample] (Maybe [Group.Group])
    deriving (Eq, Show)

data Sample = Sample
    { observedCohort :: Natural
    , observedName :: String
    , observedGroup :: String
    , observedSeed :: Integer
    , observedReward :: Natural
    , observedTokens :: Natural
    , observedTruncated :: Bool
    , observedBinding :: Invocation.Binding
    }
    deriving (Eq, Show)

data Waiting = Waiting Natural [Cohort.Task] [Replay.Pending] [Replay.Logged] [Sample]

admit :: Workload.Document -> Run -> ByteString -> Either String Report
admit expected selected encoded = do
    _ <- parseEither Json.identity (toJSON (expectedPolicy selected))
    unless (exitCode selected == 0) (Left "Evaluation process did not exit successfully")
    unless ("\n" `Bytes.isSuffixOf` encoded) (Left "Incomplete final evaluation line")
    frames <- Framing.decode encoded
    (completed, records) <- case reverse frames of
        final : previous -> pure (Framing.fields final, reverse previous)
        [] -> Left "Missing evaluation records"
    (selectedModel, count) <- parseEither (completion (expected, selected)) completed
    identities <- case selectedModel of
        Model.Materialized tokenizer base assembly -> pure (expectedPolicy selected, tokenizer, base, assembly)
        _ -> Left "Evaluation requires complete model materialization"
    let initial = [if workerMode selected == Rollout.Resident then Just (Session.start Session.Resident, Owner.start (Boundary.Owner Boundary.Inference slot)) else Nothing | slot <- [0 .. count - 1]]
    (_, owners, waiting, groups, remaining) <- foldM (cohort identities) (0, initial, [], [], records) (zip [0 ..] (Workload.cycles expected))
    rest <- foldM close remaining (reverse owners)
    unless (null rest) (Left "Output follows the final evaluation cohort or physical inference close")
    observed <- concat <$> traverse joined (reverse waiting)
    let bindings = map observedBinding observed
        distinct values = length values == Set.size (Set.fromList values)
    unless (distinct (map Invocation.boundCall bindings) && distinct (map Invocation.boundAttempt bindings) && distinct (map Invocation.boundInstance bindings)) (Left "Evaluation reused a call, attempt or instance identity")
    pure (Report (Workload.digest expected) (Artifact.hex (SHA256.hash encoded)) selected selectedModel observed (if workerMode selected == Rollout.Resident then Just (reverse groups) else Nothing))
  where
    protocol = if workerMode selected == Rollout.Batched then Session.Batched else Session.Serial
    cohort identities (offset, owners, waiting, groups, remaining) (index, workload) = do
        tasks <- traverse (task identities) (Workload.tasks workload)
        calls <- traverse (\(position, chosen) -> first show (Call.prepare (Invocation.ordinal (offset + position)) (Cohort.plan chosen))) (zip [0 ..] tasks)
        let count = length owners
            partitions = [[position | (order, position) <- zip [0 :: Int ..] (Workload.order workload), order `mod` count == slot] | slot <- [0 .. count - 1]]
        (next, pending, logged, observedGroups, rest) <- foldM (session calls) ([], [], [], [], remaining) (zip owners partitions)
        (phaseRecord, following) <- case rest of
            record : following | Fields.lookup "phase" (Framing.fields record) == Just (String "evaluation") -> pure (Framing.fields record, following)
            _ -> Left "Evaluation sessions do not end at their declared evaluation cohort"
        reported <- parseEither (parseCohort selected) (index, workload, phaseRecord)
        pure (offset + fromIntegral (length tasks), reverse next, Waiting offset tasks pending logged reported : waiting, observedGroups ++ groups, following)
    session calls (owners, pending, logged, observedGroups, remaining) (owner, selectedPositions)
        | null selectedPositions = pure (owner : owners, pending, logged, observedGroups, remaining)
        | otherwise = do
            let declaration = Session.Declaration [chosen | position <- selectedPositions, chosen <- take 1 (drop (fromIntegral position) calls)] Nothing
            case owner of
                Nothing -> do
                    (waiting, _, rest) <- first show (Replay.session protocol declaration remaining)
                    pure (Nothing : owners, pending ++ [waiting], logged, observedGroups, rest)
                Just (current, physical) -> do
                    (next, admitted, consumed, rest) <- first show (Replay.group (current, physical) declaration remaining)
                    (groupRecords, acknowledged) <- case reverse consumed of
                        final : reversed -> pure (reverse reversed, final)
                        [] -> Left "Resident group has no release acknowledgement"
                    observedGroup <- Group.observe (Owner.owner physical, Owner.groups physical, [Call.binding chosen | Session.Declaration chosenCalls _ <- [declaration], chosen <- chosenCalls]) groupRecords acknowledged
                    pure (Just next : owners, pending, logged ++ admitted, observedGroup : observedGroups, rest)
    close remaining owner = case (owner, remaining) of
        (Nothing, _) -> pure remaining
        (Just (current, physical), record : rest) -> do
            unless (Session.settled current) (Left "Resident process closes with active invocation loads")
            Owner.close physical (Framing.raw record)
            pure rest
        (Just _, []) -> Left "Missing final resident process close"
    task (policyIdentity, tokenizer, base, assembly) chosen = do
        planned <- first show (Infer.prepare (Infer.Request policyIdentity tokenizer base assembly (Workload.prompt chosen) (Workload.tokens chosen) (Workload.temperature chosen) (Workload.seed chosen)) >>= maybe Right Infer.bindPolicy (description selected))
        pure (Cohort.Task (Workload.name chosen) (Workload.group chosen) planned (Workload.rule chosen))
    joined (Waiting offset tasks pending logged reported) = do
        delimited <- first show (concat <$> traverse Replay.delimit pending)
        let admitted = map Replay.trajectory (delimited ++ logged)
            matching position = [trajectory | trajectory <- admitted, Trajectory.binding trajectory == Invocation.ordinal (offset + position)]
        ordered <- traverse (\position -> case matching position of [single] -> Right single; _ -> Left "Expected one admitted inference for each declared evaluation task") [0 .. fromIntegral (length tasks) - 1]
        scored <- either (Left . show) id (Cohort.withCohort (Cohort.Definition (expectedPolicy selected) tasks) (\declared -> first show (traverse (uncurry Cohort.record) (zip (Cohort.members declared) ordered) >>= fmap (map Cohort.reward . Cohort.observations) . Cohort.admit declared)))
        traverse check (zip3 tasks ordered scored)
      where
        check (chosen, trajectory, reward) = case [sample | sample <- reported, observedName sample == Cohort.name chosen] of
            [sample] -> do
                unless (reward == 0 || reward == 1) (Left "Expected the binary decimal-answer reward profile")
                let actual = (if reward == 0 then 0 else 1, fromIntegral (length (Trajectory.behaviorBits trajectory)), Trajectory.truncated trajectory, Infer.seed (Trajectory.request trajectory), Trajectory.binding trajectory)
                unless ((observedReward sample, observedTokens sample, observedTruncated sample, observedSeed sample, observedBinding sample) == actual) (Left "Evaluation sample differs from its admitted inference")
                pure sample
            _ -> Left "Expected one evaluation sample for each declared task"

completion :: (Workload.Document, Run) -> Object -> Parser (Model, Natural)
completion (expected, selected) fields = do
    selectedModel <- Model.binding fields
    let resident = workerMode selected == Rollout.Resident
    Json.fields (["phase", "policy", "cohorts", "tasks_sha256", "sessions"] ++ Model.fields selectedModel ++ ["worker_mode" | resident]) fields
    when resident $ do
        mode <- fields .: "worker_mode"
        unless (mode == ("resident" :: String)) (fail "Evaluation completion differs from the declared worker mode")
    phase <- fields .: "phase"
    reportedPolicy <- fields .: "policy"
    cohorts <- fields .: "cohorts" :: Parser Natural
    unless (phase == ("evaluation_complete" :: String) && reportedPolicy == expectedPolicy selected && cohorts == fromIntegral (length (Workload.cycles expected))) (fail "Missing or mismatched evaluation completion")
    identity <- fields .: "tasks_sha256"
    unless (identity == Workload.digest expected) (fail "Evaluation input identity differs from the supplied frozen bytes")
    sessions <- fields .: "sessions"
    unless (sessions > 0) (fail "Evaluation requires at least one declared session")
    pure (selectedModel, sessions)

parseCohort :: Run -> (Natural, Workload.Cycle, Object) -> Parser [Sample]
parseCohort selected (index, expected, fields) = do
    Json.fields ["phase", "cohort", "policy", "summary", "samples"] fields
    phase <- fields .: "phase"
    cohort <- fields .: "cohort"
    reportedPolicy <- fields .: "policy"
    unless (phase == ("evaluation" :: String) && cohort == index) (fail "Missing or reordered evaluation cohort")
    unless (reportedPolicy == expectedPolicy selected) (fail "Evaluation cohort policy mismatch")
    supplied <- fields .: "samples" :: Parser [Object]
    named <- traverse (\item -> (,) <$> item .: "name" <*> pure item) supplied
    let received = Map.fromList named
        declared = Workload.tasks expected
    unless (Map.size received == length named && Map.keysSet received == Set.fromList (map Workload.name declared)) (fail "Evaluation sample names differ from the declared cohort")
    observed <- traverse (checkSample received) declared
    fields .: "summary" >>= checkSummary observed
    pure observed
  where
    checkSample received task = case Map.lookup (Workload.name task) received of
        Nothing -> fail "Incomplete evaluation sample inventory"
        Just actual -> parseSample index task actual

parseSample :: Natural -> Workload.Task -> Object -> Parser Sample
parseSample index task fields = do
    Json.fields ["name", "group", "seed", "reward", "response_tokens", "truncated", "binding"] fields
    name <- fields .: "name"
    group <- fields .: "group"
    seed <- fields .: "seed"
    unless (name == Workload.name task && group == Workload.group task && seed == Workload.seed task) (fail "Evaluation sample differs from its declared task")
    reward <- fields .: "reward" >>= Json.finite
    unless (reward == 0 || reward == 1) (fail "Expected the binary decimal-answer reward profile")
    count <- fields .: "response_tokens"
    unless (count > 0 && count <= Workload.tokens task) (fail "Evaluation response length exceeds its declared budget")
    truncated <- fields .: "truncated"
    unless (not truncated || (count == Workload.tokens task && reward == 0)) (fail "Truncated evaluation sample has invalid length or reward")
    fields .: "binding" >>= withObject "evaluation binding" (Json.fields ["call", "attempt", "instance"])
    bound <- Wire.binding fields
    pure (Sample index name group seed (if reward == 0 then 0 else 1) count truncated bound)

summary :: [Sample] -> [(Key, Natural)]
summary observed =
    [ ("sample_count", fromIntegral (length observed))
    , ("reward_sum", sum (map observedReward observed))
    , ("response_tokens", sum (map observedTokens observed))
    , ("truncated_count", fromIntegral (length (filter observedTruncated observed)))
    , ("group_count", fromIntegral (Map.size groups))
    , ("zero_variance_groups", fromIntegral (length (filter ((== 1) . Set.size) (Map.elems groups))))
    ]
  where
    groups = Map.fromListWith Set.union [(observedGroup item, Set.singleton (observedReward item)) | item <- observed]

checkSummary :: [Sample] -> Object -> Parser ()
checkSummary observed fields = do
    let expected = summary observed
    Json.fields (map fst expected) fields
    mapM_ check expected
  where
    check (key, expected)
        | key == "reward_sum" = do
            actual <- fields .: key >>= Json.finite
            unless (actual == fromIntegral expected) (fail "Evaluation summary differs from its sample records")
        | otherwise = do
            actual <- fields .: key
            unless (actual == expected) (fail "Evaluation summary differs from its sample records")

sampleValue :: Sample -> Value
sampleValue sample = object (("cohort" .= observedCohort sample) : sampleFields sample)

sampleFields :: Sample -> [Pair]
sampleFields sample = ["name" .= observedName sample, "group" .= observedGroup sample, "seed" .= observedSeed sample, "reward" .= observedReward sample, "response_tokens" .= observedTokens sample, "truncated" .= observedTruncated sample, "binding" .= Wire.bindingValue (observedBinding sample)]

cohortValue :: Natural -> String -> Rollout.Batch scope -> Either String Value
cohortValue index selected batch = do
    observed <- traverse capture (Rollout.samples batch)
    pure (object ["phase" .= String "evaluation", "cohort" .= index, "policy" .= selected, "summary" .= object [key .= value | (key, value) <- summary observed], "samples" .= map (object . sampleFields) observed])
  where
    capture sample = do
        let actual = Rollout.trajectory sample
            reward = Rollout.reward sample
        unless (reward == 0 || reward == 1) (Left "Expected the binary decimal-answer reward profile")
        pure (Sample index (Rollout.name sample) (Rollout.group sample) (Infer.seed (Trajectory.request actual)) (if reward == 0 then 0 else 1) (fromIntegral (length (Trajectory.behaviorBits actual))) (Trajectory.truncated actual) (Trajectory.binding actual))

describe :: Report -> Value
describe report = object ["format" .= String "invar-evaluation-report-v1", "tasks_sha256" .= inputDigest report, "log_sha256" .= logDigest report, "policy" .= policy report, "model" .= Model.value (model report), "samples" .= map sampleValue (samples report)]

inputDigest :: Report -> String
inputDigest (Report identity _ _ _ _ _) = identity

logDigest :: Report -> String
logDigest (Report _ identity _ _ _ _) = identity

policy :: Report -> String
policy (Report _ _ selected _ _ _) = expectedPolicy selected

model :: Report -> Model
model (Report _ _ _ selected _ _) = selected

samples :: Report -> [Sample]
samples (Report _ _ _ _ observed _) = observed

sampleCohort :: Sample -> Natural
sampleCohort = observedCohort

sampleName :: Sample -> String
sampleName = observedName

sampleGroup :: Sample -> String
sampleGroup = observedGroup

sampleSeed :: Sample -> Integer
sampleSeed = observedSeed

sampleReward :: Sample -> Natural
sampleReward = observedReward

sampleTokens :: Sample -> Natural
sampleTokens = observedTokens

sampleTruncated :: Sample -> Bool
sampleTruncated = observedTruncated

residence :: Report -> Maybe [Group.Group]
residence (Report _ _ _ _ _ physical) = physical
