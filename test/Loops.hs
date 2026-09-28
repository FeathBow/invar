{-# LANGUAGE OverloadedStrings #-}

module Loops (loops) where

import BatchCalls (quote)
import Control.Monad (forM_, void)
import Data.ByteString qualified as Bytes
import Hedgehog
import Invar.Cohort qualified as C
import Invar.Infer qualified as I
import Invar.Learn qualified as L
import Invar.Learn.Worker qualified as Learner
import Invar.Loop qualified as Loop
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as R
import Invar.Schedule qualified as S
import Invar.Store qualified as Store
import Invar.Worker qualified as W
import Rollouts qualified as Fixture
import Store (workspace)
import System.Directory (createDirectory, doesPathExist, listDirectory)
import System.Exit (ExitCode (ExitFailure))
import System.FilePath ((</>))
import System.IO.Error (isAlreadyExistsError, isDoesNotExistError, tryIOError)
import System.Posix.Files (createSymbolicLink)

loops :: Group
loops = Group "Owning loop preparation" [("invalid learning settings cannot claim an output namespace", once settings), ("invalid cohorts do not advance the committed generation", once declarations), ("a different tokenizer cannot launch a rollout", once tokenizer), ("an undeclared behavior model cannot launch a rollout", once materialization), ("a selected policy description cannot change before a rollout", once description), ("declared behavior and learner models retain separate identities", once representations), ("inference launch retains its executable configuration and checkpoint", once configured), ("failed rollout keeps the checkpoint but reserves fresh identities", once failed), ("launch exceptions release preparation without reusing identities", once interrupted), ("an existing namespace cannot be opened as a fresh loop", once namespace)]
  where
    once = withTests 1 . property

setup :: FilePath -> PropertyT IO (Loop.Config, Loop.Cycle)
setup root = do
    chosen <- Fixture.options root
    let worker = R.worker chosen
        backend = Loop.Backend "unused learning executable" (W.executable worker) (W.script worker) Nothing R.Serial "unused update script" Learner.Process (W.cache worker) [[]]
        optimizer = L.Optimizer 0.002 0.8 0.95 0.0000001 0.01
        learning = L.Settings {L.policy = C.policy (R.definition chosen), L.learner = replicate 64 'b', L.reference = C.policy (R.definition chosen), L.tokenizer = replicate 64 'c', L.base = replicate 64 '0', L.assembly = replicate 64 '1', L.behaviorBase = replicate 64 'e', L.behaviorAssembly = replicate 64 'f', L.clip = 0.2, L.penalty = 0.04, L.delta = 0.0001, L.steps = 1, L.optimizer = optimizer}
    initialDescription <- evalEither (Policy.describe ("protocol-fixture", "fixture-revision") (L.policy learning, L.tokenizer learning, L.behaviorBase learning, L.behaviorAssembly learning))
    evalIO $ do
        createDirectory (root </> "input")
        Policy.stageDescription (root </> "input" </> "policy.json") initialDescription
    pure (Loop.Config backend (root </> "run") (root </> "input") (root </> "reference") learning Store.RenameExclusive, Loop.Cycle (C.tasks (R.definition chosen)) (R.order chosen) (R.delivery chosen))

ready :: Loop.Config -> Loop.Status
ready config = Loop.Ready 0 (Loop.Checkpoint (Loop.checkpoint config) (L.policy (Loop.settings config)) (L.learner (Loop.settings config)))

settings :: PropertyT IO ()
settings = do
    root <- workspace
    (config, _) <- setup root
    let invalid = config {Loop.settings = (Loop.settings config) {L.clip = 0}}
    outcome <- evalIO (Loop.withDriver invalid (const (pure ())))
    case outcome of
        Left (Loop.Settings (L.InvalidSettings _)) -> success
        unexpected -> annotateShow unexpected >> failure
    evalIO (doesPathExist (Loop.root config)) >>= (=== False)

declarations :: PropertyT IO ()
declarations = do
    root <- workspace
    (config, workload) <- setup root
    outcomes <- evalIO $ Loop.withDriver config $ \driver -> do
        empty <- void <$> Loop.run driver workload {Loop.tasks = []}
        invalid <- void <$> Loop.run driver workload {Loop.order = [0, 0]}
        badDelivery <- void <$> Loop.run driver workload {Loop.delivery = [0, 0]}
        observed <- Loop.status driver
        pure (empty, invalid, badDelivery, observed)
    outcomes === Right (Left (Loop.Rollout (R.Declaration C.EmptyCohort)), Left (Loop.Rollout (R.Scheduling (S.InvalidPermutation S.Execution))), Left (Loop.Rollout (R.Scheduling (S.InvalidPermutation S.Delivery))), ready config)
    evalIO (doesPathExist (root </> "calls")) >>= (=== False)
    evalIO (listDirectory (Loop.root config)) >>= (=== [])

tokenizer :: PropertyT IO ()
tokenizer = do
    root <- workspace
    (config, workload) <- setup root
    changed <- changeTasks (\request -> request {I.tokenizer = replicate 64 'd'}) workload
    outcomes <- evalIO $ Loop.withDriver config $ \driver -> do
        outcome <- void <$> Loop.run driver changed
        position <- Loop.status driver
        pure (outcome, position)
    outcomes === Right (Left (Loop.Plan L.TokenizerMismatch), ready config)
    evalIO (doesPathExist (root </> "calls")) >>= (=== False)
    evalIO (listDirectory (Loop.root config)) >>= (=== [])

materialization :: PropertyT IO ()
materialization = forM_ [\request -> request {I.base = replicate 64 '0'}, \request -> request {I.assembly = replicate 64 '0'}] $ \change -> do
    root <- workspace
    (config, workload) <- setup root
    changed <- changeTasks change workload
    outcomes <- evalIO $ Loop.withDriver config $ \driver -> do
        outcome <- void <$> Loop.run driver changed
        position <- Loop.status driver
        pure (outcome, position)
    outcomes === Right (Left (Loop.Plan L.MaterializationMismatch), ready config)
    evalIO (doesPathExist (root </> "calls")) >>= (=== False)
    evalIO (listDirectory (Loop.root config)) >>= (=== [])

changeTasks :: (I.Request -> I.Request) -> Loop.Cycle -> PropertyT IO Loop.Cycle
changeTasks change workload = do
    changed <- traverse bind (Loop.tasks workload)
    pure workload {Loop.tasks = changed}
  where
    bind task = do
        planned <- evalEither (I.prepare (change (I.requested (C.plan task))))
        pure task {C.plan = planned}

description :: PropertyT IO ()
description = do
    root <- workspace
    (config, workload) <- setup root
    let chosen = Loop.settings config
    changed <- evalEither (Policy.describe ("changed-model", "fixture-revision") (L.policy chosen, L.tokenizer chosen, L.behaviorBase chosen, L.behaviorAssembly chosen))
    outcomes <- evalIO $ Loop.withDriver config $ \driver -> do
        Bytes.writeFile (Loop.checkpoint config </> "policy.json") (Policy.encodeDescription changed)
        outcome <- void <$> Loop.run driver workload
        position <- Loop.status driver
        pure (outcome, position)
    outcomes === Right (Left (Loop.Policy "Selected policy description changed after publication or initial selection"), ready config)
    evalIO (doesPathExist (root </> "calls")) >>= (=== False)

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

configured :: PropertyT IO ()
configured = do
    root <- workspace
    (config, workload) <- setup root
    let script = root </> "inference.sh"
        marker = root </> "arguments"
        launch = root </> "native configuration.json"
        engine = (Loop.backend config) {Loop.inference = script, Loop.inferenceConfiguration = Just launch}
        selected = config {Loop.backend = engine}
    evalIO (writeFile script ("printf '%s\\n' \"$@\" > " ++ quote marker ++ "\nexit 7\n"))
    outcomes <- evalIO $ Loop.withDriver selected $ \driver -> do
        outcome <- void <$> Loop.run driver workload
        position <- Loop.status driver
        pure (outcome, position)
    outcomes === Right (Left (Loop.Rollout (R.Execution (W.WorkerExit (ExitFailure 7)))), ready selected)
    actual <- evalIO (readFile marker)
    lines actual === ["--cache=" ++ Loop.cache engine, "--adapter=" ++ (Loop.checkpoint config </> "adapter.safetensors"), "--config=" ++ launch]
    evalIO (listDirectory (Loop.root config)) >>= (=== [])

failed :: PropertyT IO ()
failed = do
    root <- workspace
    (config, workload) <- setup root
    outcomes <- evalIO $ Loop.withDriver config $ \driver ->
        traverse (run driver workload) [[1, 0], [0, 1]]
    observed <- evalEither outcomes
    forM_ observed $ \(outcome, position) -> do
        outcome === Left (Loop.Rollout (R.Execution (W.WorkerExit (ExitFailure 7))))
        position === ready config
    Fixture.reports root >>= (=== [(1, 1, 1), (2, 2, 2)])
    evalIO (listDirectory (Loop.root config)) >>= (=== [])
  where
    run driver workload order = do
        outcome <- void <$> Loop.run driver workload {Loop.order = order}
        position <- Loop.status driver
        pure (outcome, position)

interrupted :: PropertyT IO ()
interrupted = do
    root <- workspace
    (config, workload) <- setup root
    let executable = root </> "python"
        selected = config {Loop.backend = (Loop.backend config) {Loop.inferencePython = executable}}
    outcomes <- evalIO $ Loop.withDriver selected $ \driver -> do
        exception <- tryIOError (void <$> Loop.run driver workload)
        before <- Loop.status driver
        createSymbolicLink "/bin/sh" executable
        outcome <- void <$> Loop.run driver workload
        after <- Loop.status driver
        pure (exception, before, outcome, after)
    (exception, before, outcome, after) <- evalEither outcomes
    case exception of
        Left problem -> assert (isDoesNotExistError problem)
        Right unexpected -> annotateShow unexpected >> failure
    before === ready config
    after === ready config
    outcome === Left (Loop.Rollout (R.Execution (W.WorkerExit (ExitFailure 7))))
    Fixture.reports root >>= (=== [(2, 2, 2)])

namespace :: PropertyT IO ()
namespace = do
    root <- workspace
    (config, _) <- setup root
    evalIO (Loop.withDriver config Loop.status) >>= (=== Right (ready config))
    repeated <- evalIO (tryIOError (Loop.withDriver config (const (pure ()))))
    case repeated of
        Left problem -> assert (isAlreadyExistsError problem)
        Right unexpected -> annotateShow unexpected >> failure
