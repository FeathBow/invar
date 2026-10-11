{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Observation (Report, Source, view, source, sampled, sourceLog, sourceTrajectory, result, binding, logDigest, trajectory, policyDescription, describe, describeSource, sourcePolicy) where

import Data.Aeson (Value, object, (.=))
import Invar.Infer qualified as Infer
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Result qualified as Result
import Invar.Infer.Trajectory (Trajectory)
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Infer.Wire qualified as Wire
import Invar.Policy qualified as Policy
import Invar.Spec.Invocation qualified as V

newtype Report = Report Replay.Logged

data Source = Source String Trajectory

view :: Replay.Logged -> Report
view = Report

trajectory :: Report -> Trajectory
trajectory (Report logged) = Replay.trajectory logged

result :: Report -> Result.Result
result = Trajectory.result . trajectory

binding :: Report -> V.Binding
binding = Trajectory.binding . trajectory

logDigest :: Report -> String
logDigest (Report logged) = Replay.source logged

source :: Report -> Source
source report = Source (logDigest report) (trajectory report)

sampled :: Trajectory -> Source
sampled admitted = Source (Trajectory.digest admitted) admitted

sourceLog :: Source -> String
sourceLog (Source digest _) = digest

sourceTrajectory :: Source -> Trajectory
sourceTrajectory (Source _ admitted) = admitted

policyDescription :: Report -> Either String Policy.Description
policyDescription = sourcePolicy . source

sourcePolicy :: Source -> Either String Policy.Description
sourcePolicy (Source _ admitted) = Policy.describe (Trajectory.model admitted, Trajectory.revision admitted) (Infer.artifact requested, Infer.tokenizer requested, Infer.base requested, Infer.assembly requested)
  where
    requested = Trajectory.request admitted

describe :: Report -> Value
describe = describeSource . source

describeSource :: Source -> Value
describeSource (Source digest admitted) =
    object
        [ "log_sha256" .= digest
        , "binding" .= Wire.bindingValue (Trajectory.binding admitted)
        , "tokens" .= Trajectory.tokens admitted
        , "behavior_bits" .= Trajectory.behaviorBits admitted
        , "prompt_length" .= Trajectory.promptLength admitted
        , "text" .= Trajectory.text admitted
        , "truncated" .= Trajectory.truncated admitted
        , "model" .= Trajectory.model admitted
        , "revision" .= Trajectory.revision admitted
        , "adapter" .= Infer.artifact requested
        , "tokenizer" .= Infer.tokenizer requested
        , "base" .= Infer.base requested
        , "assembly" .= Infer.assembly requested
        , "request" .= object ["prompt" .= Infer.prompt requested, "tokens" .= Infer.tokens requested, "temperature" .= Infer.temperature requested, "seed" .= Infer.seed requested]
        ]
  where
    requested = Trajectory.request admitted
