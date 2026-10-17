module Invar.Loop (Backend (..), Config (..), CycleOf (..), Cycle, Checkpoint (..), inferenceWorker, updateWorker, scoring, stagedName, bindTasks, instantiate) where

import Data.Bifunctor (first)
import Invar.Cohort qualified as C
import Invar.Infer qualified as I
import Invar.Infer.Batch qualified as Batch
import Invar.Learn qualified as L
import Invar.Learn.Worker qualified as W
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as R
import Invar.Store qualified as S
import Invar.Worker qualified as Infer
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)
import System.FilePath ((</>))

data Backend = Backend
    { python :: FilePath
    , inferencePython :: FilePath
    , inference :: FilePath
    , inferenceConfiguration :: Maybe FilePath
    , inferenceMode :: R.Mode
    , learning :: FilePath
    , learningConfiguration :: Maybe FilePath
    , learningMode :: W.Mode
    , cache :: FilePath
    , sessions :: [[(String, String)]]
    , teacherMode :: W.Mode
    }
data Config = Config {backend :: Backend, root :: FilePath, checkpoint :: FilePath, reference :: FilePath, settings :: L.Settings, publication :: S.Method}
data CycleOf grade = Cycle {tasks :: [C.TaskOf grade], order :: [Natural], delivery :: [Natural]}
type Cycle = CycleOf C.Graded
data Checkpoint = Checkpoint {directory :: FilePath, policy :: String, learner :: String}
    deriving (Eq, Show)

bindTasks :: Policy.Description -> [C.TaskOf grade] -> Either String [C.TaskOf grade]
bindTasks description = traverse bind
  where
    bind task = do
        planned <- first show (I.bindPolicy description (C.plan task))
        pure task {C.plan = planned}

instantiate :: (C.Grade grade) => (String, String, String, String) -> Workload.CycleOf grade -> Either String (CycleOf grade)
instantiate (policy', tokenizer, base, assembly) workload = do
    tasks' <- traverse prepare (Workload.tasks workload)
    first show (C.withCohort (C.Definition policy' tasks') (const ()))
    pure (Cycle tasks' (Workload.order workload) (Workload.delivery workload))
  where
    prepare sample = do
        planned <- first show (I.prepare I.Request {I.artifact = policy', I.tokenizer = tokenizer, I.base = base, I.assembly = assembly, I.prompt = Workload.prompt sample, I.tokens = Workload.tokens sample, I.temperature = Workload.temperature sample, I.seed = Workload.seed sample})
        pure (C.Task (Workload.name sample) planned (Workload.grade sample))

scoring :: Config -> Checkpoint -> Maybe Batch.Reference
scoring config selected
    | L.reference (settings config) == policy selected = Nothing
    | otherwise = Just (Batch.Reference (reference config) (L.reference (settings config)))

inferenceWorker :: Config -> Checkpoint -> Infer.Worker
inferenceWorker config selected =
    Infer.Worker (inferencePython engine) (inference engine) (cache engine) (directory selected </> "adapter.safetensors") [] (inferenceConfiguration engine)
  where
    engine = backend config

updateWorker :: Config -> (Checkpoint, Natural) -> W.Worker
updateWorker config (selected, ordinal) =
    W.Worker
        { W.executable = python (backend config)
        , W.script = learning (backend config)
        , W.cache = cache (backend config)
        , W.checkpoint = directory selected
        , W.reference = reference config
        , W.output = root config </> stagedName ordinal
        , W.configuration = learningConfiguration (backend config)
        }

stagedName :: Natural -> String
stagedName ordinal = "staging" ++ show ordinal
