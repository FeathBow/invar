{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Cohort (Checked, admit, admitLog, describe, inferences, update, rewards) where

import Control.Monad (unless)
import Data.Aeson (Object, Value (..), object, toJSON, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Set qualified as Set
import Data.Text.Encoding (decodeUtf8)
import Invar.Cohort qualified as Cohort
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Observation qualified as Observation
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Learn qualified as Learn
import Invar.Learn.Observed qualified as Input
import Invar.Learn.Report qualified as Report
import Invar.Learn.Request qualified as Request
import Invar.Spec.Invocation qualified as V
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)

data Checked = Checked [Observation.Report] Report.Report [Rational]

admitLog :: Learn.Settings -> (Workload.Cycle, Natural) -> ByteString -> Either String Checked
admitLog settings (workload, selected) encoded = do
    observed <- observeInferences settings (workload, selected) encoded
    reported <- Report.admit selected encoded
    admit settings (workload, observed) reported

observeInferences :: Learn.Settings -> (Workload.Cycle, Natural) -> ByteString -> Either String [Observation.Report]
observeInferences settings (workload, selected) encoded = do
    let tasks = Workload.tasks workload
        count = fromIntegral (length tasks)
    unless (selected >= count) (Left "Update call cannot precede its declared cohort")
    frames <- Framing.decode encoded
    grouped <- Framing.groups frames
    let events = [(Framing.raw frame, Framing.fields frame) | frame <- frames]
    traverse (inference events grouped) (zip [selected - count ..] tasks)
  where
    inference events grouped (index, selectedTask) = do
        declared <- Input.task settings selectedTask
        let bound = V.Binding (V.CallId index) (V.AttemptId index) (V.Instance index)
            matches fields = Fields.lookup "binding" fields == Just (Wire.bindingValue bound)
            groups = [group | group <- grouped, any (matches . Framing.fields . Framing.consumed) (Framing.members group)]
            serial = [fields | (_, fields) <- events, Fields.lookup "stage" fields == Just (String "loaded_adapter"), matches fields]
        case (groups, serial) of
            ([], [_]) -> inferenceSegment bound events >>= Observation.admit (Cohort.plan declared) bound
            ([group], []) -> Observation.admitGroup (Cohort.plan declared) bound group
            _ -> Left "Expected one serial or batched inference for the declared cohort member"

inferenceSegment :: V.Binding -> [(ByteString, Object)] -> Either String ByteString
inferenceSegment bound events = do
    let matches (_, fields) = Fields.lookup "stage" fields == Just (String "loaded_adapter") && Fields.lookup "binding" fields == Just (Wire.bindingValue bound)
    unless (length (filter matches events) == 1) (Left "Expected one bound inference load for the declared cohort member")
    let (_, remaining) = break matches events
        (prefix, finished) = break (\(_, fields) -> Fields.lookup "stage" fields == Just (String "result")) remaining
    case finished of
        result : _ -> pure (Bytes.unlines (map fst (prefix ++ [result])))
        [] -> Left "History inference has no completed result"

admit :: Learn.Settings -> (Workload.Cycle, [Observation.Report]) -> Report.Report -> Either String Checked
admit settings (workload, observed) reported = do
    updated <- parseEither (withObject "update invocation" Wire.binding) (Report.invocation reported)
    let bindings = updated : map Observation.binding observed
    unless (distinct (map V.boundCall bindings) && distinct (map V.boundAttempt bindings) && distinct (map V.boundInstance bindings)) (Left "History cohort reused an invocation identity")
    (program, payload, scored) <- numericalInput settings (workload, observed)
    expected <- Json.decode payload >>= parseEither Request.parse
    actual <- parseEither Request.parse (Report.request reported)
    reportedProgram <- parseEither (withObject "update invocation" (.: "program")) (Report.invocation reported)
    unless (reportedProgram == decodeUtf8 program && Request.logical actual == Request.logical expected) (Left "Consumed update differs from its declared tasks and actual inference observations")
    pure (Checked observed reported scored)
  where
    distinct values = length values == Set.size (Set.fromList values)

numericalInput :: Learn.Settings -> (Workload.Cycle, [Observation.Report]) -> Either String (ByteString, ByteString, [Rational])
numericalInput settings (workload, observed) = Input.input settings workload (map Observation.result observed)

inferences :: Checked -> [Observation.Report]
inferences (Checked observed _ _) = observed

update :: Checked -> Report.Report
update (Checked _ reported _) = reported

rewards :: Checked -> [Rational]
rewards (Checked _ _ values) = values

describe :: Checked -> Value
describe checked = object ["log_sha256" .= Report.logDigest (update checked), "inferences" .= map Observation.describe (inferences checked), "rewards" .= map (toJSON . (fromRational :: Rational -> Double)) (rewards checked), "update" .= Report.describe (update checked)]
