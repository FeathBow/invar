{-# LANGUAGE RoleAnnotations #-}

module Invar.Worker.Resident (Options (..), Resident, withResident, withBorrowed, run) where

import Control.Monad (unless)
import Data.Bifunctor (first)
import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import Invar.Infer.Batch qualified as Batch
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Session qualified as Session
import Invar.Infer.Trajectory (Trajectory)
import Invar.Process qualified as Process
import Invar.Process.Resident qualified as ProcessResident
import Invar.Resident qualified as Boundary
import Invar.Transcript qualified as Transcript
import Invar.Worker qualified as Worker
import Numeric.Natural (Natural)

data Options = Options {worker :: Worker.Worker, owner :: Natural, transcript :: Transcript.Transcript}

type role Resident nominal
data Resident scope = Resident (ProcessResident.Resident scope) (IORef Session.Session)

withResident :: Options -> (forall scope. Resident scope -> IO (Either Worker.Failure value)) -> IO (Either Worker.Failure value)
withResident options action = do
    session <- newIORef (Session.start Session.Resident)
    let selected = worker options
        arguments = [Worker.script selected, "--cache=" ++ Worker.cache selected, "--session=" ++ show (owner options)] ++ maybe [] (\path -> ["--config=" ++ path]) (Worker.configuration selected)
        launch = Process.Launch (Worker.executable selected) arguments (Worker.environment selected) (transcript options)
        identity = Boundary.Owner Boundary.Inference (owner options)
        ready _ = do
            current <- readIORef session
            pure (if Session.settled current then Right () else Left (Call.Protocol "Resident process closes with active invocation loads"))
    first failure <$> ProcessResident.withResident launch identity ready (\process -> first transport <$> action (Resident process session))

withBorrowed :: ProcessResident.Resident scope -> (Resident scope -> IO value) -> IO value
withBorrowed process action = do
    session <- newIORef (Session.start Session.Resident)
    returned <- action (Resident process session)
    current <- readIORef session
    unless (Session.settled current) (ioError (userError "Borrowed inference role exits with active invocation loads"))
    pure returned

run :: Resident scope -> FilePath -> Maybe Batch.Reference -> [Call.Call] -> IO (Either Worker.Failure [Trajectory])
run _ _ _ [] = pure (Right [])
run (Resident process session) adapter reference calls = do
    returned <- ProcessResident.hosted process $ \channel physical -> do
        current <- readIORef session
        exchanged <- Worker.exchange channel (\(Session.Request selected) -> Batch.input adapter reference selected) current (Session.Hosted physical (Session.Declaration calls (fmap Batch.identity reference)))
        case exchanged of
            Left problem -> pure (Left (transport (Worker.failure problem)))
            Right (following, trajectories, Just updated) -> writeIORef session following >> pure (Right (trajectories, updated))
            Right (_, _, Nothing) -> pure (Left (Process.Protocol "Resident group was admitted without its release"))
    pure (first failure returned)

failure :: Process.Failure Call.Error -> Worker.Failure
failure (Process.Exit status) = Worker.WorkerExit status
failure (Process.Rejected problem) = Worker.InvalidOutput problem
failure (Process.Protocol problem) = Worker.ProtocolFailure problem

transport :: Worker.Failure -> Process.Failure Call.Error
transport (Worker.WorkerExit status) = Process.Exit status
transport (Worker.InvalidOutput problem) = Process.Rejected problem
transport (Worker.ProtocolFailure problem) = Process.Protocol problem
