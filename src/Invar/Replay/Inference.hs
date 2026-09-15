{-# LANGUAGE OverloadedStrings #-}

module Invar.Replay.Inference (Mode (..), Reference, Call, admit, calls, ownerCalls, evaluation, describe, decodeCalls, decodeResidentCalls, observe, observeResident) where

import Control.Monad (foldM, unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value (..), object, parseJSON, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Invar.Artifact qualified as Artifact
import Invar.Evaluation qualified as Evaluation
import Invar.Infer.Framing (Frame (..))
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Measurement.Duration qualified as Duration
import Invar.Replay.Call (Call)
import Invar.Replay.Call qualified as Call
import Invar.Replay.Load qualified as Load
import Invar.Replay.Resident qualified as ResidentReplay
import Invar.Resident qualified as Owner
import Invar.Resident.Inference qualified as Physical
import Invar.Resident.Observation qualified as Residence
import Invar.Spec.Invocation qualified as V
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)

data Mode = Process | Session | Batched | Resident deriving (Eq, Show)
data Reference = Reference Evaluation.Report [Call] (Maybe [(Natural, [Call])])

admit :: Workload.Document -> (Evaluation.Run, Mode) -> ByteString -> Either String Reference
admit tasks (run, mode) encoded = do
    evaluated <- Evaluation.admit tasks run encoded
    frames <- stream encoded
    (consumed, results) <- inventory 0 frames
    let samples = Evaluation.samples evaluated
        declared = Map.fromList [(V.boundCall (Evaluation.sampleBinding sample), sample) | sample <- samples]
        named = Map.fromList [((index, Workload.name task), task) | (index, workload) <- zip [0 ..] (Workload.cycles tasks), task <- Workload.tasks workload]
    unless (length consumed == length samples && length results == length samples) (Left "Incomplete reference worker inventory")
    planned <- traverse (referenceCall (declared, named, Evaluation.policy evaluated)) (zip consumed results)
    unless (distinctBindings planned) (Left "Repeated reference consumption binding")
    when (mode /= Process) (mapM_ Call.session planned)
    owners <- physicalCalls evaluated planned
    when (mode == Resident && maybe True null owners) (Left "Resident replay requires a complete resident evaluation reference")
    pure (Reference evaluated planned owners)

physicalCalls :: Evaluation.Report -> [Call] -> Either String (Maybe [(Natural, [Call])])
physicalCalls evaluated planned = case Evaluation.residence evaluated of
    Nothing -> pure Nothing
    Just physical -> Just <$> traverse (selected physical) [0 .. Physical.sessions physical - 1]
  where
    indexed = Map.fromList [(V.boundCall (Call.bound call), call) | call <- planned]
    selected physical slot = do
        let bindings = concat [Residence.bindings group | (_, observed) <- Physical.groups physical, group <- observed, Residence.physicalOwner group == Owner.Owner Owner.Inference slot]
        members <- traverse match bindings
        pure (slot, members)
    match bound = do
        call <- maybe (Left "Resident owner invocation has no reference call") pure (Map.lookup (V.boundCall bound) indexed)
        unless (Call.bound call == bound) (Left "Resident owner reference binding differs from its physical group")
        pure call

referenceCall :: (Map.Map V.CallId Evaluation.Sample, Map.Map (Natural, String) Workload.Task, String) -> ((Int, Frame), (Int, Frame)) -> Either String Call
referenceCall (declared, named, policy) ((firstPosition, Frame first fields), (lastPosition, Frame lastOutput _)) = do
    unless (firstPosition < lastPosition) (Left "Reference result precedes its consumption")
    binding <- parseEither Wire.binding fields
    sample <- maybe (Left "Reference consumption has no declared evaluation sample") Right (Map.lookup (V.boundCall binding) declared)
    unless (binding == Evaluation.sampleBinding sample) (Left "Reference consumption binding differs from its evaluation sample")
    task <- maybe (Left "Reference sample has no frozen task") Right (Map.lookup (Evaluation.sampleCohort sample, Evaluation.sampleName sample) named)
    let requested = object ["prompt" .= Workload.prompt task, "seed" .= Workload.seed task, "tokens" .= Workload.tokens task, "temperature" .= Workload.temperature task]
    unless (Fields.lookup "request" fields == Just requested) (Left "Reference consumption differs from the frozen task")
    unless (Fields.lookup "adapter" fields == Just (toJSON policy)) (Left "Reference policy mismatch")
    Call.admit (Evaluation.sampleCohort sample) (first, lastOutput)

stream :: ByteString -> Either String [Frame]
stream = Framing.decode

inventory :: Int -> [Frame] -> Either String ([(Int, Frame)], [(Int, Frame)])
inventory _ [] = pure ([], [])
inventory index records@(first : rest)
    | Framing.grouped first = do
        (group, remaining) <- Framing.takeGroup records
        (consumed, results) <- inventory (index + 3) remaining
        pure ([(index, Framing.consumed member) | member <- Framing.members group] ++ consumed, [(index + 2, Framing.result member) | member <- Framing.members group] ++ results)
    | otherwise = do
        (consumed, results) <- inventory (index + 1) rest
        let include expected values = if Fields.lookup "stage" (Framing.fields first) == Just (String expected) then (index, first) : values else values
        pure (include "consumed" consumed, include "result" results)

decodeCalls :: ByteString -> Either String [Call]
decodeCalls = decodeWith False

decodeResidentCalls :: ByteString -> Either String [Call]
decodeResidentCalls = decodeWith True

decodeWith :: Bool -> ByteString -> Either String [Call]
decodeWith allowEmpty encoded = do
    supplied <- Json.decode encoded >>= parseEither parseJSON
    planned <- traverse Call.decode supplied
    unless ((allowEmpty || not (null planned)) && distinctBindings planned) (Left (if allowEmpty then "Expected distinct resident replay calls" else "Expected distinct nonempty replay calls"))
    pure planned

distinctBindings :: [Call] -> Bool
distinctBindings planned = distinct (map (V.boundCall . Call.bound) planned) && distinct (map (V.boundAttempt . Call.bound) planned) && distinct (map (V.boundInstance . Call.bound) planned)
  where
    distinct values = length values == Set.size (Set.fromList values)

observe :: (Mode, Int) -> [Call] -> ByteString -> Either String Value
observe (Resident, _) _ _ = Left "Resident replay output requires an explicit physical owner"
observe (mode, exitCode) planned encoded = do
    unless (exitCode == 0) (Left "Direct worker process did not exit successfully")
    unless (not (null planned) && (mode /= Process || length planned == 1)) (Left "Expected one process replay call or a nonempty session")
    when (mode /= Process) (mapM_ Call.session planned)
    frames <- stream encoded
    let (prefix, execution) = span (\(Frame _ fields) -> Fields.lookup "stage" fields `elem` map (Just . String) ["loading", "profile", "load"]) frames
        stages = [stage | Frame _ fields <- prefix, Just (String stage) <- [Fields.lookup "stage" fields]]
    unless (stages `elem` [["load"], ["profile", "load"], ["loading", "load"], ["loading", "profile", "load"]] && not (any (\(Frame _ fields) -> Fields.member "phase" fields) prefix)) (Left "Expected one ordered model load in a direct replay")
    loading <- case reverse prefix of
        Frame raw fields : _ -> Duration.admit raw fields
        [] -> Left "Missing direct model load"
    let digest = Artifact.hex (SHA256.hash encoded)
    if mode == Batched
        then observeBatch (digest, loading) planned execution
        else do
            (observations, remaining, _) <- foldM step ([], execution, Nothing) (zip [0 :: Natural ..] planned)
            unless (null remaining) (Left "Trailing direct worker output")
            case (mode, reverse observations) of
                (Process, [observed]) -> pure (Object (Fields.union (Fields.fromList ["stdout_sha256" .= digest, "load" .= Duration.value loading]) (Fields.delete "index" (Fields.delete "binding" observed))))
                (Session, observed) -> pure (object ["stdout_sha256" .= digest, "load" .= Duration.value loading, "calls" .= map Object observed])
                _ -> Left "Incomplete direct replay"
  where
    step (observed, frames, previous) (index, call) = do
        afterUnload <- case frames of
            Frame _ fields : rest | Fields.lookup "stage" fields == Just (String "unloaded_adapter") -> do
                preceding <- maybe (Left "Adapter replacement before the first replay result") Right previous
                parseEither (Load.unloaded preceding) fields
                pure rest
            _ -> pure frames
        let (loading, execution) = case afterUnload of
                frame@(Frame _ fields) : rest | Fields.lookup "stage" fields == Just (String "loaded_adapter") -> ([frame], rest)
                _ -> ([], afterUnload)
        unless (all (\(Frame _ fields) -> not (Fields.member "phase" fields)) loading) (Left "Unexpected phase inside a direct replay")
        mapM_ (\(Frame _ fields) -> parseEither (Load.admit call) fields) loading
        case execution of
            Frame first consumed : Frame rawDuration timing : Frame lastOutput result : rest -> do
                unless (map (Fields.lookup "stage") [consumed, timing, result] == map (Just . String) ["consumed", "inference", "result"] && not (any (Fields.member "phase") [consumed, timing, result])) (Left "Missing or reordered direct replay stages")
                unless (consumed == Call.consumed call) (Left "Direct worker consumed different input")
                actual <- Call.admit (Call.cohort call) (first, lastOutput)
                measured <- Duration.admit rawDuration timing
                let row = Fields.fromList ["index" .= index, "binding" .= Fields.lookup "binding" consumed, "result_equal" .= (Call.result actual == Call.result call), "response_tokens" .= Call.responseTokens actual, "inference" .= Duration.value measured]
                pure (row : observed, rest, Just actual)
            _ -> Left "Incomplete direct worker stages"

observeBatch :: (String, Duration.Duration) -> [Call] -> [Frame] -> Either String Value
observeBatch (digest, loading) planned frames = do
    (group, remaining) <- Framing.takeGroup frames
    unless (null remaining && length planned == length (Framing.members group)) (Left "Direct finite batch differs from its complete declared inventory")
    rows <- traverse observed (zip3 [0 :: Natural ..] planned (Framing.members group))
    let Frame raw fields = Framing.duration group
    measured <- Duration.admit raw fields
    pure (object ["stdout_sha256" .= digest, "load" .= Duration.value loading, "inference" .= Duration.value measured, "calls" .= rows])
  where
    observed (index, call, member) = do
        _ <- parseEither (Load.admit call) (Framing.fields (Framing.loaded member))
        let consumed = Framing.consumed member
        unless (Framing.fields consumed == Call.consumed call) (Left "Direct batch consumed different input or order")
        actual <- Call.admit (Call.cohort call) (Framing.raw consumed, Framing.raw (Framing.result member))
        pure (object ["index" .= index, "binding" .= Fields.lookup "binding" (Call.consumed call), "result_equal" .= (Call.result actual == Call.result call), "response_tokens" .= Call.responseTokens actual])

calls :: Reference -> [Call]
calls (Reference _ planned _) = planned

ownerCalls :: Reference -> Maybe [(Natural, [Call])]
ownerCalls (Reference _ _ owners) = owners

evaluation :: Reference -> Evaluation.Report
evaluation (Reference evaluated _ _) = evaluated

describe :: Reference -> Value
describe (Reference evaluated planned owners) = object (["reference_log_sha256" .= Evaluation.logDigest evaluated, "tasks_sha256" .= Evaluation.inputDigest evaluated, "calls" .= map Call.value planned, "scope" .= ("complete evaluation and replay input/result correspondence; not measurement-profile admission or execution permission" :: Text)] ++ maybe [] physical owners)
  where
    physical selected = ["residence" .= object ["sessions" .= length selected, "owners" .= [object ["owner" .= owner, "calls" .= map Call.value members] | (owner, members) <- selected]]]

observeResident :: (Natural, Int) -> [Call] -> ByteString -> Either String Value
observeResident selected planned = fmap ResidentReplay.describe . ResidentReplay.admit selected planned
