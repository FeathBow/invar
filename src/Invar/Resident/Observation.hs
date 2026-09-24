{-# LANGUAGE OverloadedStrings #-}

module Invar.Resident.Observation (State, Group, empty, owner, count, inference, learning, finish, source, prefix, body, modelProfiles, physicalOwner, ordinal, bindings, acknowledgement, loading, activation, release, describe) where

import Control.Monad (foldM, unless, when)
import Data.Aeson (Object, Value (..), object, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8)
import Invar.Infer.Framing (stageName)
import Invar.Infer.Framing qualified as Frame
import Invar.Infer.Wire qualified as Wire
import Invar.Learn qualified as Learn
import Invar.Learn.Framing qualified as Learner
import Invar.Learn.Report qualified as Report
import Invar.Learn.Trace qualified as Trace
import Invar.Measurement.Duration qualified as Duration
import Invar.Replay.Call qualified as Call
import Invar.Replay.Load qualified as Load
import Invar.Resident qualified as Boundary
import Invar.Spec.Invocation qualified as Invocation
import Numeric.Natural (Natural)

data State = State
    { owner :: Boundary.Owner
    , count :: Natural
    , originalPrefix :: [Frame.Frame]
    , model :: Maybe Object
    , identities :: Identities
    , clock :: Maybe Duration.Clock
    }

data Group = Group
    { physicalOwner :: Boundary.Owner
    , ordinal :: Natural
    , prefix :: [Frame.Frame]
    , body :: [Frame.Frame]
    , modelProfiles :: [Frame.Frame]
    , bindings :: [Invocation.Binding]
    , acknowledgement :: Frame.Frame
    , loading :: Maybe Duration.Duration
    , activation :: Maybe Duration.Duration
    , release :: Duration.Duration
    , costs :: [Duration.Duration]
    }
    deriving (Eq, Show)

empty :: Boundary.Owner -> State
empty selected = State selected 0 [] Nothing (Set.empty, Set.empty, Set.empty) Nothing

type Identities = (Set.Set Invocation.CallId, Set.Set Invocation.AttemptId, Set.Set Invocation.Instance)

fresh :: Identities -> [Invocation.Binding] -> Either String Identities
fresh = foldM admit
  where
    admit (calls, attempts, instances) bound = do
        let call = Invocation.boundCall bound
            attempt = Invocation.boundAttempt bound
            instanceId = Invocation.boundInstance bound
        when (Set.member call calls || Set.member attempt attempts || Set.member instanceId instances) (Left "Resident physical process reuses a historical invocation identity")
        pure (Set.insert call calls, Set.insert attempt attempts, Set.insert instanceId instances)

inference :: (Natural, State) -> [Frame.Frame] -> Either String (State, Group, [Frame.Frame])
inference (cohort, current) records = do
    requireRole Boundary.Inference current
    (leading, remaining) <- preparation Boundary.Inference current records
    (group, rest) <- Frame.takeGroup remaining
    calls <- traverse (\member -> Call.admit cohort (Frame.raw (Frame.consumed member), Frame.raw (Frame.result member))) (Frame.members group)
    mapM_ Call.session calls
    loaded <- traverse (\(call, member) -> parseEither (Load.admit call) (Frame.fields (Frame.loaded member))) (zip calls (Frame.members group))
    execution <- Frame.decode (Frame.source group)
    complete current (leading, execution, rest) (loaded, map (Frame.fields . Frame.consumed) (Frame.members group))

learning :: State -> (Learn.Settings, Report.Report) -> [Frame.Frame] -> Either String (State, Group, [Frame.Frame])
learning current (settings, report) = learner (Trace.validate settings) current report

learner :: (Report.Report -> [Object] -> Either String ()) -> State -> Report.Report -> [Frame.Frame] -> Either String (State, Group, [Frame.Frame])
learner validate current report records = do
    requireRole Boundary.Learning current
    (leading, remaining) <- preparation Boundary.Learning current records
    (execution, rest) <- throughResult remaining
    let raw = Frame.encode (leading ++ execution)
        (ready, _) = break ((== Just (String "consumed")) . stageName) execution
    consumed <- case drop (length ready) execution of
        value : _ -> pure value
        [] -> Left "Missing resident learner consumption"
    Learner.readiness (count current == 0) (Report.request report) (Frame.encode (leading ++ ready ++ [consumed]))
    Learner.completion (Report.request report) raw
    validate report (map Frame.fields execution)
    case execution of
        loaded : _ -> complete current (leading, execution, rest) ([Frame.fields loaded], [Frame.fields consumed])
        [] -> Left "Missing resident learner execution"

requireRole :: Boundary.Role -> State -> Either String ()
requireRole expected current = case owner current of
    Boundary.Owner actual _ -> unless (actual == expected || actual == Boundary.Shared) (Left "Resident observation has a different numerical owner role")

preparation :: Boundary.Role -> State -> [Frame.Frame] -> Either String ([Frame.Frame], [Frame.Frame])
preparation role current records = do
    let (leading, rest) = span (\record -> stageName record `elem` map (Just . String) ["loading", "profile", "load", "activation"]) records
        initial = case role of
            Boundary.Inference -> [["load"], ["loading", "profile", "load"]]
            _ -> [["load", "activation"], ["loading", "profile", "load", "activation"]]
        expected = if count current == 0 then initial else [["activation"]]
    unless (map stageName leading `elem` map (map (Just . String)) expected) (Left "Resident observation requires one actual initial load and subsequent activation costs")
    mapM_ (\record -> when (Fields.member "phase" (Frame.fields record)) (Left "Unexpected phase within a resident group")) leading
    pure (leading, rest)

throughResult :: [Frame.Frame] -> Either String ([Frame.Frame], [Frame.Frame])
throughResult records = case break ((== Just (String "result")) . stageName) records of
    (preceding, result : rest) -> pure (preceding ++ [result], rest)
    _ -> Left "Incomplete resident update result"

complete :: State -> ([Frame.Frame], [Frame.Frame], [Frame.Frame]) -> ([Object], [Object]) -> Either String (State, Group, [Frame.Frame])
complete current (leading, execution, remaining) (loaded, consumed) = do
    (ack, rest) <- case remaining of
        value : rest -> pure (value, rest)
        [] -> Left "Resident group has no release acknowledgement"
    let retained = if count current == 0 then leading else originalPrefix current
        selectedModel = case (model current, loaded) of
            (Nothing, value : _) -> Just value
            (previous, _) -> previous
        observedModel = current {model = selectedModel}
    mapM_ (correspondence observedModel retained) loaded
    received <- traverse (parseEither Wire.binding) consumed
    when (null received) (Left "Resident group requires a consumed invocation")
    admitted <- fresh (identities current) received
    loads <- traverse (parseEither (.: "load")) consumed
    retired <- Boundary.observeRelease (owner current, loads, Frame.encode (leading ++ execution)) (Frame.raw ack)
    initialLoad <- measurement "load" leading
    selected <- measurement "activation" leading
    measured <- traverse timing (filter (\record -> stageName record `elem` map (Just . String) ["load", "activation", "inference", "probability_roles", "reward_update", "artifacts", "checkpoint"]) (leading ++ execution))
    let durations = measured ++ [retired]
    selectedClock <- clocks (clock current) durations
    let next = observedModel {count = count current + 1, originalPrefix = retained, identities = admitted, clock = selectedClock}
        profiles = filter ((== Just (String "profile")) . stageName) retained
    pure (next, Group (owner current) (count current) leading execution profiles received ack initialLoad selected retired durations, rest)

correspondence :: State -> [Frame.Frame] -> Object -> Either String ()
correspondence current retained actual = do
    let profiles = [Frame.fields record | record <- retained, stageName record == Just (String "profile")]
        expected = profiles ++ maybe [] pure (model current)
    mapM_ (\fields -> unless (all (\key -> Fields.lookup key fields == Fields.lookup key actual) ["model", "revision"]) (Left "Resident model or revision differs from its original physical load")) expected

measurement :: Text -> [Frame.Frame] -> Either String (Maybe Duration.Duration)
measurement name records = case filter ((== Just (String name)) . stageName) records of
    [] -> pure Nothing
    [record] -> Just <$> timing record
    _ -> Left "Repeated resident operation measurement"

clocks :: Maybe Duration.Clock -> [Duration.Duration] -> Either String (Maybe Duration.Clock)
clocks expected measured = case maybe [] pure expected ++ map Duration.clock measured of
    [] -> pure Nothing
    selected : rest -> do
        unless (all (== selected) rest) (Left "Resident physical process mixes measurement clocks")
        pure (Just selected)

finish :: State -> [Frame.Frame] -> Either String (Frame.Frame, Duration.Duration, [Frame.Frame])
finish current (record : rest) = do
    elapsed <- Boundary.closed (owner current) (count current) (Frame.raw record)
    _ <- clocks (clock current) [elapsed]
    pure (record, elapsed, rest)
finish _ [] = Left "Missing final resident process close"

timing :: Frame.Frame -> Either String Duration.Duration
timing record = Duration.admit (Frame.raw record) (Frame.fields record)

source :: Group -> ByteString
source group = Frame.encode (prefix group ++ body group)

describe :: Group -> Value
describe group = object ["owner" .= ownerValue, "group" .= ordinal group, "source_jsonl" .= decodeUtf8 (source group), "released_json" .= decodeUtf8 (Frame.raw (acknowledgement group)), "load" .= loading group, "activation" .= activation group, "release" .= release group, "costs" .= costs group]
  where
    ownerValue = Boundary.ownerValue (physicalOwner group)
