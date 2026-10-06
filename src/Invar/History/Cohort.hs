{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Cohort (Checked, admit, describe, inferences, update, rewards) where

import Control.Monad (unless)
import Data.Aeson (Value, object, toJSON, withObject, (.:), (.=))
import Data.Aeson.Types (parseEither)
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

data Checked = Checked [Observation.Report] Report.Report [Rational]

admit :: Learn.Settings -> ([Cohort.Task], [Replay.Logged]) -> Report.Report -> Either String Checked
admit settings (tasks, logged) reported = do
    updated <- parseEither (withObject "update invocation" Wire.binding) (Report.invocation reported)
    let observed = map Observation.view logged
        bindings = updated : map Observation.binding observed
    unless (distinct (map V.boundCall bindings) && distinct (map V.boundAttempt bindings) && distinct (map V.boundInstance bindings)) (Left "History cohort reused an invocation identity")
    (program, payload, scored) <- Input.input settings tasks (map Observation.trajectory observed)
    reportedProgram <- parseEither (withObject "update invocation" (.: "program")) (Report.invocation reported)
    unless (reportedProgram == decodeUtf8 program) (Left Report.recordedElsewhere)
    expected <- Json.decode payload >>= parseEither Request.parse
    actual <- parseEither Request.parse (Report.request reported)
    unless (Request.logical actual == Request.logical expected) (Left "Consumed update differs from its declared tasks and actual inference observations")
    pure (Checked observed reported scored)
  where
    distinct values = length values == Set.size (Set.fromList values)

inferences :: Checked -> [Observation.Report]
inferences (Checked observed _ _) = observed

update :: Checked -> Report.Report
update (Checked _ reported _) = reported

rewards :: Checked -> [Rational]
rewards (Checked _ _ values) = values

describe :: Checked -> Value
describe checked = object ["log_sha256" .= Report.logDigest (update checked), "inferences" .= map Observation.describe (inferences checked), "rewards" .= map (toJSON . (fromRational :: Rational -> Double)) (rewards checked), "update" .= Report.describe (update checked)]
