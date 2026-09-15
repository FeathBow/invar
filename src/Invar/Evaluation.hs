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
    sampleBinding,
) where

import Control.Monad (foldM, unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Object, Value (..), object, toJSON, withObject, (.:), (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Pair, Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Invar.Artifact qualified as Artifact
import Invar.Infer qualified as Infer
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Model (Model (..))
import Invar.Infer.Model qualified as Model
import Invar.Infer.Result qualified as Result
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Resident.Inference qualified as Resident
import Invar.Rollout qualified as Rollout
import Invar.Spec.Invocation qualified as Invocation
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)

data Run = Run {expectedPolicy :: String, exitCode :: Int}
    deriving (Eq, Show)

data Report = Report String String Run Model [Sample] (Maybe Resident.Ledger)
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

workerStages :: [Value]
workerStages = map String ["loading", "profile", "load", "unloaded_adapter", "loaded_adapter", "consumed", "inference", "result"]

admit :: Workload.Document -> Run -> ByteString -> Either String Report
admit expected selected encoded = do
    _ <- parseEither Json.identity (toJSON (expectedPolicy selected))
    unless (exitCode selected == 0) (Left "Evaluation process did not exit successfully")
    unless ("\n" `Bytes.isSuffixOf` encoded) (Left "Incomplete final evaluation line")
    frames <- Framing.decode encoded
    grouped <- Framing.groups frames
    case reverse frames of
        [] -> Left "Missing evaluation records"
        completedFrame : previous -> do
            let completed = Framing.fields completedFrame
            (selectedModel, observed) <- parseEither (const (document (expected, selected) (map Framing.fields (reverse previous)) completed)) Null
            resident <- parseEither residentMode completed
            physical <-
                if resident
                    then do
                        owners <- parseEither (.: "sessions") completed
                        Just <$> Resident.admit (expected, expectedPolicy selected, selectedModel) owners (reverse previous)
                    else pure Nothing
            mapM_ (parseEither (checkModel selectedModel) . Framing.fields) [record | group <- grouped, member <- Framing.members group, record <- [Framing.loaded member, Framing.consumed member, Framing.result member]]
            pure (Report (Workload.digest expected) (Artifact.hex (SHA256.hash encoded)) selected selectedModel observed physical)

document :: (Workload.Document, Run) -> [Object] -> Object -> Parser (Model, [Sample])
document (expected, selected) records completed = do
    selectedModel <- Model.binding completed
    mapM_ (checkModel selectedModel) (completed : records)
    resident <- residentMode completed
    (cohorts, loads, pending) <- foldM (collect resident) ([], [], 0) records
    unless (pending == 0) (fail "Model load recorded outside every evaluation cohort")
    let declared = Workload.cycles expected
        ordered = reverse cohorts
    unless (length ordered == length declared) (fail "Incomplete or trailing evaluation records")
    checkComplete (expected, selected, selectedModel) (reverse loads) completed
    observed <- concat <$> traverse (parseCohort selected) (zip3 [0 ..] declared ordered)
    let bindings = map observedBinding observed
        distinct values = length values == Set.size (Set.fromList values)
    unless (distinct (map Invocation.boundCall bindings) && distinct (map Invocation.boundAttempt bindings) && distinct (map Invocation.boundInstance bindings)) (fail "Evaluation reused a call, attempt or instance identity")
    pure (selectedModel, observed)

collect :: Bool -> ([Object], [Natural], Natural) -> Object -> Parser ([Object], [Natural], Natural)
collect resident (cohorts, loads, current) record
    | Fields.member "phase" record = do
        unless (Fields.lookup "phase" record == Just (String "evaluation") && not (Fields.member "stage" record)) (fail "Unexpected evaluation record")
        pure (record : cohorts, current : loads, 0)
    | otherwise = do
        stage <- record .: "stage"
        unless (stage `elem` (workerStages ++ [String extra | resident, extra <- ["activation", "released", "closed"]])) (fail "Unknown worker record in evaluation output")
        pure (cohorts, loads, current + if stage == String "load" then 1 else 0)

checkModel :: Model -> Object -> Parser ()
checkModel expected record = when (not (Framing.grouped (Framing.Frame "" record)) && Fields.lookup "stage" record `elem` map (Just . String) ["loaded_adapter", "consumed", "result"]) $ do
    actual <- Model.binding record
    unless (actual == expected) (fail "Evaluation model and tokenizer bindings disagree")

checkComplete :: (Workload.Document, Run, Model) -> [Natural] -> Object -> Parser ()
checkComplete (expected, selected, selectedModel) loads fields = do
    resident <- residentMode fields
    let declaredSessions = Fields.member "sessions" fields
    when (resident && not declaredSessions) (fail "Resident evaluation requires declared physical owners")
    Json.fields (["phase", "policy", "cohorts", "tasks_sha256"] ++ Model.fields selectedModel ++ ["sessions" | declaredSessions] ++ ["worker_mode" | resident]) fields
    phase <- fields .: "phase"
    reportedPolicy <- fields .: "policy"
    cohorts <- fields .: "cohorts" :: Parser Natural
    unless (phase == ("evaluation_complete" :: String) && reportedPolicy == expectedPolicy selected && cohorts == fromIntegral (length (Workload.cycles expected))) (fail "Missing or mismatched evaluation completion")
    identity <- fields .: "tasks_sha256"
    unless (identity == Workload.digest expected) (fail "Evaluation input identity differs from the supplied frozen bytes")
    when declaredSessions $ do
        sessions <- fields .: "sessions"
        unless (sessions > 0 && (resident || all (== sessions) loads)) (fail "Declared session count differs from the recorded model loads of a cohort")

residentMode :: Object -> Parser Bool
residentMode fields = case Fields.lookup "worker_mode" fields of
    Nothing -> pure False
    Just (String "resident") -> pure True
    _ -> fail "Unsupported declared evaluation worker mode"

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
        let actual = Rollout.observation sample
            reward = Rollout.reward sample
        unless (reward == 0 || reward == 1) (Left "Expected the binary decimal-answer reward profile")
        pure (Sample index (Rollout.name sample) (Rollout.group sample) (Infer.seed (Result.consumed actual)) (if reward == 0 then 0 else 1) (fromIntegral (length (Result.behaviorBits actual))) (Result.truncated actual) (Invocation.completedBinding (Rollout.completion sample)))

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

sampleBinding :: Sample -> Invocation.Binding
sampleBinding = observedBinding

residence :: Report -> Maybe Resident.Ledger
residence (Report _ _ _ _ _ physical) = physical
