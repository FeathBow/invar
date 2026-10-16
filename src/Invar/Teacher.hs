{-# LANGUAGE OverloadedStrings #-}

module Invar.Teacher (prepare, reading, joined) where

import Control.Monad (unless)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Map.Strict qualified as Map
import Data.Text qualified as Text
import Invar.Artifact qualified as Artifact
import Invar.Canonical qualified as Canonical
import Invar.Float32 qualified as Float32
import Invar.Infer qualified as Infer
import Invar.Infer.Observation qualified as Inference
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Learn qualified as Learn
import Invar.Learn.Algorithm qualified as Algorithm
import Invar.Policy qualified as Policy
import Invar.Score qualified as Score
import Invar.Spec.Invocation qualified as V

target :: Algorithm.Teacher -> Either String Score.Target
target chosen = do
    let identity = (Algorithm.teacherAdapter chosen, Algorithm.teacherTokenizer chosen, Algorithm.teacherBase chosen, Algorithm.teacherAssembly chosen)
    description <- Policy.describe (Algorithm.teacherModel chosen, Algorithm.teacherRevision chosen) identity
    pure (Score.Target identity (Just description))

prepare :: Algorithm.Teacher -> Inference.Source -> Either String Score.Plan
prepare chosen source = do
    Algorithm.teacher chosen
    unless (Algorithm.teacherTemperature chosen == Infer.temperature requested) (Left "The teacher scores at the temperature the student sampled at")
    selected <- target chosen
    either (Left . show) Right (Score.prepare source selected)
  where
    requested = Trajectory.request (Inference.sourceTrajectory source)

reading :: Score.Report -> Either String Learn.Reading
reading scored = do
    unless (all Float32.logProbability taught) (Left "Every teacher word must be a finite nonpositive FP32 log probability")
    pure (Learn.Reading (V.completedBinding (Score.completion scored)) taught)
  where
    taught = Score.probabilityWords scored

joined :: String -> Learn.Reading -> String
joined trajectory taught =
    Artifact.hex . SHA256.hash . Canonical.encode $
        object
            [ ("format", Canonical.Text "invar-distilled-result-v1")
            , ("trajectory", Canonical.Text (Text.pack trajectory))
            , ("teacher", object [("call", binding (Learn.readingCall taught)), ("words", Canonical.Array [Canonical.Integer (toInteger word) | word <- Learn.readingWords taught])])
            ]
  where
    object = Canonical.Object . Map.fromList
    binding (V.Binding (V.CallId call) (V.AttemptId attempt) (V.Instance instanceId)) = object [("call", natural call), ("attempt", natural attempt), ("instance", natural instanceId)]
    natural = Canonical.Integer . toInteger
