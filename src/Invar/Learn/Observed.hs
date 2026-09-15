module Invar.Learn.Observed (task, input) where

import Control.Monad (unless)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Invar.Cohort qualified as Cohort
import Invar.Infer qualified as Infer
import Invar.Infer.Result qualified as Result
import Invar.Learn qualified as Learn
import Invar.Workload qualified as Workload

-- Share reward scoring and numerical lowering across history admission and
-- native observations. Neither path creates a live invocation.
input :: Learn.Settings -> Workload.Cycle -> [Result.Result] -> Either String (ByteString, ByteString, [Rational])
input settings workload observed = do
    let tasks = Workload.tasks workload
    unless (length tasks == length observed) (Left "Inference inventory differs from its declared cohort")
    declared <- traverse (task settings) tasks
    either
        (Left . show)
        id
        ( Cohort.withCohort (Cohort.Definition (Learn.policy settings) declared) $ \cohort -> do
            samples <- first show (traverse (uncurry Cohort.record) (zip (Cohort.members cohort) observed))
            batch <- first show (Cohort.admit cohort samples)
            (program, payload) <- first show (Learn.observedInput settings batch)
            pure (program, payload, map Cohort.reward (Cohort.observations batch))
        )

task :: Learn.Settings -> Workload.Task -> Either String Cohort.Task
task settings selected = do
    let requested = Infer.Request (Learn.policy settings) (Learn.tokenizer settings) (Learn.behaviorBase settings) (Learn.behaviorAssembly settings) (Workload.prompt selected) (Workload.tokens selected) (Workload.temperature selected) (Workload.seed selected)
    planned <- first show (Infer.prepare requested)
    pure (Cohort.Task (Workload.name selected) (Workload.group selected) planned (Workload.rule selected))
