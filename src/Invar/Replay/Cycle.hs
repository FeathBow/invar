{-# LANGUAGE OverloadedStrings #-}

module Invar.Replay.Cycle (Reference, admit, describe, prepareInput, inspect) where

import Control.Monad (unless)
import Data.Aeson (Value (..), object, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.Maybe (isJust)
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8)
import Invar.History.Cohort qualified as Cohort
import Invar.History.Publication qualified as Publication
import Invar.History.Trace qualified as Trace
import Invar.Infer.Framing qualified as Frame
import Invar.Infer.Observation qualified as Inference
import Invar.Json qualified as Json
import Invar.Learn qualified as Learn
import Invar.Learn.Report qualified as Report
import Invar.Learn.Request qualified as Request
import Invar.Policy qualified as Policy
import Invar.Replay.Call qualified as Call
import Invar.Replay.Update qualified as Update
import Invar.Replay.Update.Artifacts qualified as Artifacts
import Invar.Resident qualified as Boundary
import Invar.Resident.Observation qualified as Resident
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)

data Reference = Reference Trace.Run Workload.Document Trace.Checked [Update.Update]

admit :: (FilePath -> IO Bool) -> (FilePath, Trace.Run) -> (Workload.Document, ByteString) -> IO Reference
admit directory (initial, run) (tasks, encoded) = do
    let modes = (Trace.inferenceMode run, Trace.learningMode run)
    select <- case modes of
        (Trace.Resident, Trace.Resident) -> pure Update.admitResident
        (Trace.Shared, Trace.Shared) -> pure Update.admitShared
        _ -> invalid "Complete-cycle replay requires the declared resident or shared physical lifetimes"
    checked <- either invalid pure (Trace.admit run tasks encoded)
    updates <- select directory (Update.Run initial (fromIntegral (length (Workload.cycles tasks))) (Trace.exitCode run)) encoded
    pure (Reference run tasks checked (Update.updates updates))

prepareInput :: (Learn.Settings, Update.Update) -> (Workload.Cycle, Natural) -> ByteString -> Either String Value
prepareInput (chosen, expected) selection encoded = do
    (program, payload) <- Cohort.prepareInput chosen selection encoded
    numerical <- Json.decode payload
    actual <- parseEither Request.parse numerical
    reference <- parseEither Request.parse (Report.request (Update.report expected))
    expectedProgram <- parseEither (withObject "reference update invocation" (.: "program")) (Report.invocation (Update.report expected))
    let same = decodeUtf8 program == expectedProgram && Request.logical actual == Request.logical reference
    pure (object ["program" .= decodeUtf8 program, "request" .= numerical, "equal_to_reference" .= same])

describe :: Reference -> Either String Value
describe (Reference run _ checked updates) = do
    cycles <- traverse generation (zip3 [0 ..] (Trace.generations checked) updates)
    pure (object ["format" .= ("invar-cycle-replay-v1" :: Text), "mode" .= mode, "sessions" .= Trace.sessions run, "publication" .= Trace.method run, "cycles" .= cycles, "close" .= Trace.closing checked])
  where
    mode = if Trace.inferenceMode run == Trace.Shared then "shared" else "resident" :: Text
    generation (index, observed, update) = do
        groups <- traverse (group index) (Trace.residentGroups observed)
        current <- settingsFor (Trace.settings run) (Cohort.update (Trace.cohort observed))
        selected <- Publication.source observed >>= Json.decode . Policy.encodeDescription
        pure (object ["index" .= index, "settings" .= settings current, "policy" .= selected, "groups" .= groups, "update" .= Update.value update, "publication" .= Trace.publication observed])

group :: Natural -> Resident.Group -> Either String Value
group index observed = do
    let body = Resident.body observed
        inference = case body of
            first : _ -> Frame.grouped first
            [] -> False
    calls <-
        if inference
            then do
                (batch, _) <- Frame.takeGroup body
                traverse (\member -> Call.admit index (Frame.raw (Frame.consumed member), Frame.raw (Frame.result member))) (Frame.members batch)
            else pure []
    pure (object ["kind" .= (if inference then "inference" else "learning" :: Text), "owner" .= Boundary.ownerValue (Resident.physicalOwner observed), "group" .= Resident.ordinal observed, "calls" .= map Call.value calls])

settingsFor :: Learn.Settings -> Report.Report -> Either String Learn.Settings
settingsFor original observed = do
    (policy, learner) <- parseEither (withObject "consumed update settings" (\fields -> (,) <$> fields .: "policy" <*> fields .: "learner")) (Report.request observed)
    pure original {Learn.policy, Learn.learner}

settings :: Learn.Settings -> Value
settings chosen =
    object
        [ "policy" .= Learn.policy chosen
        , "learner" .= Learn.learner chosen
        , "reference-digest" .= Learn.reference chosen
        , "tokenizer-digest" .= Learn.tokenizer chosen
        , "base-digest" .= Learn.base chosen
        , "assembly-digest" .= Learn.assembly chosen
        , "behavior-base-digest" .= Learn.behaviorBase chosen
        , "behavior-assembly-digest" .= Learn.behaviorAssembly chosen
        , "clip" .= show (Learn.clip chosen)
        , "penalty" .= show (Learn.penalty chosen)
        , "delta" .= show (Learn.delta chosen)
        , "rate" .= show (Learn.learningRate optimizer)
        , "beta1" .= show (Learn.firstMoment optimizer)
        , "beta2" .= show (Learn.secondMoment optimizer)
        , "optimizer-epsilon" .= show (Learn.epsilon optimizer)
        , "decay" .= show (Learn.weightDecay optimizer)
        ]
  where
    optimizer = Learn.optimizer chosen

inspect :: Reference -> (FilePath, Int, ByteString) -> IO Value
inspect (Reference run tasks expected _) (directory, status, encoded) = do
    actual <- either invalid pure (Trace.admit run {Trace.output = directory, Trace.exitCode = status} tasks encoded)
    compared <- traverse compareGeneration (zip (Trace.generations expected) (Trace.generations actual))
    decisions <- traverse (either invalid pure . parseEither (withObject "cycle comparison" (.: "equal"))) compared
    pure (object ["equal" .= and decisions, "cycles" .= compared, "actual" .= Trace.describe actual, "scope" .= ("complete direct numerical cycle observations with actual publication entries; no live Invar invocation or qualification authority" :: Text)])

compareGeneration :: (Trace.Generation, Trace.Generation) -> IO Value
compareGeneration (expected, actual) = do
    previous <- Publication.observe expected
    published <- Publication.observe actual
    let first = Cohort.update (Trace.cohort expected)
        second = Cohort.update (Trace.cohort actual)
        inferSame = map Inference.result (Cohort.inferences (Trace.cohort expected)) == map Inference.result (Cohort.inferences (Trace.cohort actual))
        profiles observed = [(Resident.physicalOwner group_, map Frame.fields (Resident.modelProfiles group_)) | group_ <- Trace.residentGroups observed]
        physical observed = [(Resident.physicalOwner group_, Resident.ordinal group_, Resident.bindings group_, isJust (Resident.loading group_)) | group_ <- Trace.residentGroups observed]
        profileSame = profiles expected == profiles actual
        physicalSame = physical expected == physical actual
    inputSame <- either invalid pure (Report.sameInput first second)
    fields <- Artifacts.compare (second, Publication.directory published) (first, Publication.directory previous)
    updateSame <- either invalid pure (parseEither (.: "result_equal") (Fields.fromList fields))
    unless physicalSame (invalid "Direct cycle physical owner groups or load counts differ from the reference")
    pure (object ["equal" .= and [inferSame, inputSame, updateSame, profileSame], "inferences_equal" .= inferSame, "inputs_equal" .= inputSame, "profiles_equal" .= profileSame, "physical_lifetime_equal" .= physicalSame, "update" .= object fields, "publication" .= Publication.describe published])

invalid :: String -> IO value
invalid = ioError . userError
