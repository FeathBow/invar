{-# LANGUAGE OverloadedStrings #-}

module Loops (loops) where

import Control.Monad (forM_)
import Data.Aeson (Value (Null))
import Hedgehog
import Invar.Cohort qualified as C
import Invar.Infer qualified as I
import Invar.Learn qualified as L
import Invar.Learn.Worker qualified as Learner
import Invar.Loop qualified as Loop
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as R
import Invar.Runtime qualified as Runtime
import Invar.Store qualified as Store
import Invar.Worker qualified as W
import LearnerFixture qualified as F
import Rollouts qualified as Fixture
import Sessions qualified
import Store (workspace)
import System.Directory (createDirectory, doesPathExist)
import System.FilePath ((</>))

loops :: Group
loops = Group "Training run preparation" [("an initial description that differs from the declared materialization is refused before the run starts", once initial), ("a run that selects shared execution for one role only, with positive staleness, with roles that differ in executable, script or model cache, or with more than one session, is refused before it claims its output", once sharing), ("declared behavior and learner models retain separate identities", once representations)]
  where
    once = withTests 1 . property

setup :: FilePath -> PropertyT IO (Loop.Config, Loop.Cycle)
setup root = do
    chosen <- Fixture.options root
    let worker = R.worker chosen
        backend = Loop.Backend "unused learning executable" (W.executable worker) (W.script worker) Nothing R.Serial "unused update script" Nothing Learner.Process (W.cache worker) [[]] Learner.Process
        learning = F.configured
    initialDescription <- evalEither (Policy.describe ("protocol-fixture", "fixture-revision") (L.policy learning, L.tokenizer learning, L.behaviorBase learning, L.behaviorAssembly learning))
    evalIO $ do
        createDirectory (root </> "input")
        Policy.stageDescription (root </> "input" </> "policy.json") initialDescription
    pure (Loop.Config backend (root </> "run") (root </> "input") (root </> "reference") learning Store.RenameExclusive, Loop.Cycle (C.tasks (R.definition chosen)) (R.order chosen) (R.delivery chosen))

initial :: PropertyT IO ()
initial = do
    root <- workspace
    (config, _) <- setup root
    let changed = config {Loop.settings = (Loop.settings config) {L.behaviorBase = replicate 64 'd'}}
        problem = "Initial policy description differs from the declared inference materialization"
    document <- evalEither Sessions.workload
    concurrent <- evalIO (Runtime.run (Runtime.Run changed 0 document) Null)
    case concurrent of
        Left (Runtime.Declaration refused) -> refused === problem
        unexpected -> annotateShow unexpected >> failure
    evalIO (doesPathExist (Loop.root config)) >>= (=== False)

sharing :: PropertyT IO ()
sharing = do
    root <- workspace
    (config, _) <- setup root
    document <- evalEither Sessions.workload
    let modes inference learning = config {Loop.backend = (Loop.backend config) {Loop.inferenceMode = inference, Loop.learningMode = learning}}
        joint = modes R.Shared Learner.Shared
        engine = Loop.backend joint
        aligned count = joint {Loop.backend = engine {Loop.python = Loop.inferencePython engine, Loop.learning = Loop.inference engine, Loop.sessions = replicate count []}}
    forM_ [(modes R.Shared Learner.Resident, 0, "Both numerical roles must explicitly select shared execution"), (modes R.Resident Learner.Shared, 0, "Both numerical roles must explicitly select shared execution"), (joint, 1, "Shared execution requires staleness zero"), (joint, 0, "Shared roles require the same executable, script and model cache"), (aligned 2, 0, "Shared execution requires exactly one physical session")] $ \(chosen, lag, problem) -> do
        refused <- evalIO (Runtime.run (Runtime.Run chosen lag document) Null)
        case refused of
            Left (Runtime.Declaration found) -> found === problem
            unexpected -> annotateShow unexpected >> failure
    evalIO (doesPathExist (Loop.root config)) >>= (=== False)

representations :: PropertyT IO ()
representations = do
    root <- workspace
    (config, workload) <- setup root
    let chosen = Loop.settings config
    forM_ (Loop.tasks workload) $ \task -> do
        let request = I.requested (C.plan task)
        L.materialization chosen request === Right ()
        assert (L.base chosen /= I.base request && L.assembly chosen /= I.assembly request)
        L.materialization chosen request {I.base = L.base chosen} === Left L.MaterializationMismatch
        L.materialization chosen request {I.assembly = L.assembly chosen} === Left L.MaterializationMismatch
