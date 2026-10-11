module Invar.Teacher (prepare) where

import Control.Monad (unless)
import Invar.Infer qualified as Infer
import Invar.Infer.Observation qualified as Inference
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Learn.Algorithm qualified as Algorithm
import Invar.Policy qualified as Policy
import Invar.Score qualified as Score

target :: Algorithm.Teacher -> Either String Score.Target
target chosen = do
    let identity = (Algorithm.teacherAdapter chosen, Algorithm.teacherTokenizer chosen, Algorithm.teacherBase chosen, Algorithm.teacherAssembly chosen)
    description <- Policy.describe (Algorithm.teacherModel chosen, Algorithm.teacherRevision chosen) identity
    pure (Score.Target identity (Just description))

prepare :: Algorithm.Teacher -> Inference.Source -> Either String Score.Plan
prepare chosen source = do
    Algorithm.teacher (Infer.tokenizer requested) chosen
    unless (Algorithm.teacherTemperature chosen == Infer.temperature requested) (Left "The teacher scores at the temperature the student sampled at")
    selected <- target chosen
    either (Left . show) Right (Score.prepare source selected)
  where
    requested = Trajectory.request (Inference.sourceTrajectory source)
