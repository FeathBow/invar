module Invar.Learn.Observed (task, input, distilled) where

import Control.Monad (unless)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.Map.Strict (Map)
import Invar.Cohort qualified as Cohort
import Invar.Infer qualified as Infer
import Invar.Infer.Trajectory (Trajectory)
import Invar.Learn qualified as Learn
import Invar.Policy qualified as Policy
import Invar.Workload qualified as Workload

input :: Learn.Settings -> [Cohort.Task] -> [Trajectory] -> Either String (ByteString, ByteString, [Rational])
input settings declared observed = admitted settings declared observed $ \batch -> do
    (program, payload) <- first show (Learn.observedInput settings batch)
    pure (program, payload, map Cohort.reward (Cohort.observations batch))

distilled :: Learn.Settings -> [Cohort.TaskOf Cohort.Ungraded] -> [Trajectory] -> Map String Learn.Reading -> Either String (ByteString, ByteString)
distilled settings declared observed readings = admitted settings declared observed (first show . Learn.observedDistilled settings readings)

admitted :: (Cohort.Grade grade) => Learn.Settings -> [Cohort.TaskOf grade] -> [Trajectory] -> (forall scope. Cohort.BatchOf scope grade -> Either String result) -> Either String result
admitted settings declared observed continue = do
    unless (length declared == length observed) (Left "Inference inventory differs from its declared cohort")
    either
        (Left . show)
        id
        ( Cohort.withCohort (Cohort.Definition (Learn.behaviorPolicy (Learn.schedule settings)) declared) $ \cohort -> do
            samples <- first show (traverse (uncurry Cohort.record) (zip (Cohort.members cohort) observed))
            batch <- first show (Cohort.admit cohort samples)
            continue batch
        )

task :: Learn.Settings -> Maybe Policy.Description -> Workload.Task -> Either String Cohort.Task
task settings description selected = do
    let requested = Infer.Request (Learn.behaviorPolicy (Learn.schedule settings)) (Learn.tokenizer settings) (Learn.behaviorBase settings) (Learn.behaviorAssembly settings) (Workload.prompt selected) (Workload.tokens selected) (Workload.temperature selected) (Workload.seed selected)
    planned <- first show (Infer.prepare requested >>= maybe Right Infer.bindPolicy description)
    pure (Cohort.Task (Workload.name selected) planned (Workload.grade selected))
