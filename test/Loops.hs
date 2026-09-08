{-# LANGUAGE OverloadedStrings #-}

module Loops (loops) where

import Control.Monad (forM_, void)
import Hedgehog
import Invar.Cohort qualified as C
import Invar.Learn qualified as L
import Invar.Loop qualified as Loop
import Invar.Rollout qualified as R
import Invar.Schedule qualified as S
import Invar.Store qualified as Store
import Invar.Worker qualified as W
import Rollouts qualified as Fixture
import Store (workspace)
import System.Directory (doesPathExist, listDirectory)
import System.Exit (ExitCode (ExitFailure))
import System.FilePath ((</>))
import System.IO.Error (isAlreadyExistsError, isDoesNotExistError, tryIOError)
import System.Posix.Files (createSymbolicLink)

loops :: Group
loops = Group "Owning loop preparation" [("invalid learning settings cannot claim an output namespace", once settings), ("invalid cohorts do not advance the committed generation", once declarations), ("a different tokenizer cannot launch a rollout", once tokenizer), ("a different materialized model cannot launch a rollout", once materialization), ("failed rollout keeps the checkpoint but reserves fresh identities", once failed), ("launch exceptions release preparation without reusing identities", once interrupted), ("an existing namespace cannot be opened as a fresh loop", once namespace)]
  where
    once = withTests 1 . property

setup :: FilePath -> PropertyT IO (Loop.Config, Loop.Cycle)
setup root = do
    chosen <- Fixture.options root
    let worker = R.worker chosen
        backend = Loop.Backend (W.executable worker) (W.script worker) "unused update script" (W.cache worker) [[]]
        optimizer = L.Optimizer 0.002 0.8 0.95 0.0000001 0.01
        learning = L.Settings (C.policy (R.definition chosen)) (replicate 64 'b') (replicate 64 'c') (replicate 64 'c') (replicate 64 'e') (replicate 64 'f') 0.2 0.04 0.0001 optimizer
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
    let changed = config {Loop.settings = (Loop.settings config) {L.tokenizer = replicate 64 'd'}}
    outcomes <- evalIO $ Loop.withDriver changed $ \driver -> do
        outcome <- void <$> Loop.run driver workload
        position <- Loop.status driver
        pure (outcome, position)
    outcomes === Right (Left (Loop.Plan L.TokenizerMismatch), ready changed)
    evalIO (doesPathExist (root </> "calls")) >>= (=== False)
    evalIO (listDirectory (Loop.root config)) >>= (=== [])

materialization :: PropertyT IO ()
materialization = forM_ [\chosen -> chosen {L.base = replicate 64 '0'}, \chosen -> chosen {L.assembly = replicate 64 '0'}] $ \change -> do
    root <- workspace
    (config, workload) <- setup root
    let changed = config {Loop.settings = change (Loop.settings config)}
    outcomes <- evalIO $ Loop.withDriver changed $ \driver -> do
        outcome <- void <$> Loop.run driver workload
        position <- Loop.status driver
        pure (outcome, position)
    outcomes === Right (Left (Loop.Plan L.MaterializationMismatch), ready changed)
    evalIO (doesPathExist (root </> "calls")) >>= (=== False)
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
        selected = config {Loop.backend = (Loop.backend config) {Loop.python = executable}}
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
