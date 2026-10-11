{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Cohort (Checked, Inputs (..), admit, describe, inferences, update) where

import Control.Monad (unless)
import Data.Aeson (Value, object, toJSON, withObject, (.:), (.=))
import Data.Aeson.Types (Pair, parseEither)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text.Encoding (decodeUtf8)
import Invar.Cohort qualified as Cohort
import Invar.Infer.Observation qualified as Observation
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Learn qualified as Learn
import Invar.Learn.Observed qualified as Input
import Invar.Learn.Report qualified as Report
import Invar.Learn.Request qualified as Request
import Invar.Spec.Invocation qualified as V

data Inputs = Rewarded [Cohort.Task] [Replay.Logged] | Distilled [Cohort.TaskOf Cohort.Ungraded] [Replay.Logged] (Map String Learn.Reading)

data Signals = Rewards [Rational] | Readings [Learn.Reading]

data Checked = Checked [Observation.Report] Report.Report Signals

admit :: Learn.Settings -> Inputs -> Report.Report -> Either String Checked
admit settings inputs reported = do
    updated <- parseEither (withObject "update invocation" Wire.binding) (Report.invocation reported)
    let observed = map Observation.view logged
        trajectories = map Observation.trajectory observed
        bindings = updated : map Observation.binding observed
    unless (distinct (map V.boundCall bindings) && distinct (map V.boundAttempt bindings) && distinct (map V.boundInstance bindings)) (Left "History cohort reused an invocation identity")
    (program, payload, signals) <- case inputs of
        Rewarded tasks _ -> (\(program, payload, scored) -> (program, payload, Rewards scored)) <$> Input.input settings tasks trajectories
        Distilled tasks _ readings -> (\(program, payload) -> (program, payload, Readings [reading | task <- tasks, Just reading <- [Map.lookup (Cohort.name task) readings]])) <$> Input.distilled settings tasks trajectories readings
    reportedProgram <- parseEither (withObject "update invocation" (.: "program")) (Report.invocation reported)
    unless (reportedProgram == decodeUtf8 program) (Left Report.recordedElsewhere)
    expected <- Json.decode payload >>= parseEither Request.parse
    actual <- parseEither Request.parse (Report.request reported)
    unless (Request.logical actual == Request.logical expected) (Left "Consumed update differs from its declared tasks and actual inference observations")
    pure (Checked observed reported signals)
  where
    logged = case inputs of
        Rewarded _ found -> found
        Distilled _ found _ -> found
    distinct values = length values == Set.size (Set.fromList values)

inferences :: Checked -> [Observation.Report]
inferences (Checked observed _ _) = observed

update :: Checked -> Report.Report
update (Checked _ reported _) = reported

describe :: Checked -> Value
describe checked@(Checked _ _ signals) = object (["log_sha256" .= Report.logDigest (update checked), "inferences" .= map Observation.describe (inferences checked), "update" .= Report.describe (update checked)] ++ described signals)
  where
    described :: Signals -> [Pair]
    described (Rewards values) = ["rewards" .= map (toJSON . (fromRational :: Rational -> Double)) values]
    described (Readings read') = ["teacher_readings" .= [object ["call" .= Wire.bindingValue (Learn.readingCall reading), "words" .= Learn.readingWords reading] | reading <- read']]
