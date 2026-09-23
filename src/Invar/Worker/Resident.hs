{-# LANGUAGE RoleAnnotations #-}

module Invar.Worker.Resident (Options (..), Resident, Receipt, withResident, withBorrowed, run, report, completion, loaded, acknowledgement, session) where

import Control.Exception (bracket, mask_)
import Control.Monad (unless, void)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.IORef (IORef, modifyIORef', newIORef, readIORef, writeIORef)
import Invar.Infer.Batch qualified as Batch
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Result qualified as Result
import Invar.Process qualified as Process
import Invar.Process.Resident qualified as ProcessResident
import Invar.Resident qualified as Boundary
import Invar.Spec.Invocation qualified as Invocation
import Invar.Spec.Load qualified as Load
import Invar.Worker qualified as Worker
import Numeric.Natural (Natural)

data Options = Options {worker :: Worker.Worker, owner :: Natural, echo :: ByteString -> IO ()}
data State = State Load.Registry Natural
data Receipt = Receipt Invocation.Completion Result.Result Load.Fact ByteString Natural
data Progress = Awaiting | Consumed Batch.Permit | Completed [(Invocation.Completion, Result.Result, Load.Fact)] Boundary.Release | Released [Receipt]

type role Resident nominal
data Resident scope = Resident (ProcessResident.Resident scope) (IORef State) Boundary.Owner Natural

withResident :: Options -> (forall scope. Resident scope -> IO (Either Worker.Failure value)) -> IO (Either Worker.Failure value)
withResident options action = bracket (newIORef (State Load.empty 0)) retireOwner $ \state -> do
    let selected = worker options
        arguments = [Worker.script selected, "--cache=" ++ Worker.cache selected, "--session=" ++ show (owner options)] ++ maybe [] (\path -> ["--config=" ++ path]) (Worker.configuration selected)
        launch = Process.Launch (Worker.executable selected) arguments (Worker.environment selected) (echo options)
        identity = Boundary.Owner Boundary.Inference (owner options)
        closing = ProcessResident.Handshake (Boundary.close identity) (close identity state)
    first failure <$> ProcessResident.withResident launch (const closing) (\process -> first transport <$> action (Resident process state identity (owner options)))
  where
    retireOwner state = modifyIORef' state (\(State registry groups) -> State (Load.close registry) groups)
    close identity state encoded = do
        State registry groups <- readIORef state
        pure $ first Call.Protocol $ do
            unless (null (Load.active registry)) (Left "Resident process closes with active invocation loads")
            void (Boundary.closed identity groups encoded)

withBorrowed :: ProcessResident.Resident scope -> Boundary.Owner -> (Resident scope -> IO value) -> IO value
withBorrowed process identity@(Boundary.Owner _ index) action = bracket (newIORef (State Load.empty 0)) retire $ \state -> do
    returned <- action (Resident process state identity index)
    State registry _ <- readIORef state
    unless (null (Load.active registry)) (ioError (userError "Borrowed inference role exits with active invocation loads"))
    pure returned
  where
    retire state = modifyIORef' state (\(State registry count) -> State (Load.close registry) count)

run :: Resident scope -> FilePath -> [Call.Call] -> IO (Either Worker.Failure [Receipt])
run _ _ [] = pure (Right [])
run (Resident process state identity index) adapter calls = do
    progress <- newIORef Awaiting
    let exchange = Process.Exchange (Batch.input adapter calls) (authorize process (state, progress) calls) (complete identity progress)
        transaction = ProcessResident.Transaction exchange (release (state, progress) index)
    returned <- ProcessResident.exchange process transaction
    case first failure returned of
        Left problem -> pure (Left problem)
        Right _ -> do
            finished <- readIORef progress
            pure $ case finished of
                Released values -> Right values
                _ -> Left (Worker.ProtocolFailure "Resident result has no checked release acknowledgement")

authorize :: ProcessResident.Resident scope -> (IORef State, IORef Progress) -> [Call.Call] -> ByteString -> IO (Either Call.Error ByteString)
authorize process (state, progress) calls encoded = mask_ $ do
    current <- readIORef progress
    State registry groups <- readIORef state
    physical <- ProcessResident.groups process
    case current of
        Awaiting -> case (if physical == 0 then Batch.authorize else Batch.authorizeActivation) registry calls encoded of
            Left problem -> pure (Left problem)
            Right (updated, permit) -> do
                writeIORef state (State updated groups)
                writeIORef progress (Consumed permit)
                pure (Right (Batch.permission permit))
        _ -> pure (Left (Call.Protocol "Resident invocation already holds consumption permits"))

complete :: Boundary.Owner -> IORef Progress -> ByteString -> IO (Either Call.Error ())
complete identity progress encoded = mask_ $ do
    current <- readIORef progress
    case current of
        Consumed permit -> case observed permit of
            Left problem -> pure (Left problem)
            Right (values, prepared) -> writeIORef progress (Completed values prepared) >> pure (Right ())
        _ -> pure (Left (Call.Protocol "Resident result has no accepted consumption permits"))
  where
    observed permit = do
        values <- Batch.observe permit encoded
        prepared <- first Call.Protocol (Boundary.prepare identity [fact | (_, _, fact) <- values] encoded)
        pure (values, prepared)

release :: (IORef State, IORef Progress) -> Natural -> ByteString -> IO (Either Call.Error (ProcessResident.Handshake Call.Error))
release (state, progress) index _ = do
    current <- readIORef progress
    pure $ case current of
        Completed values prepared -> Right (ProcessResident.Handshake (Boundary.request prepared) (acknowledge prepared values))
        _ -> Left (Call.Protocol "Resident release precedes a checked invocation result")
  where
    acknowledge prepared values encoded = mask_ $ do
        State registry groups <- readIORef state
        case first Call.Protocol (Boundary.retire prepared registry encoded) of
            Left problem -> pure (Left problem)
            Right updated -> do
                let completed = [Receipt finished result fact encoded index | (finished, result, fact) <- values]
                writeIORef state (State updated (groups + 1))
                writeIORef progress (Released completed)
                pure (Right ())

report :: Receipt -> Result.Result
report (Receipt _ result _ _ _) = result

completion :: Receipt -> Invocation.Completion
completion (Receipt completed _ _ _ _) = completed

loaded :: Receipt -> Load.Fact
loaded (Receipt _ _ fact _ _) = fact

acknowledgement :: Receipt -> ByteString
acknowledgement (Receipt _ _ _ encoded _) = encoded

session :: Receipt -> Natural
session (Receipt _ _ _ _ index) = index

failure :: Process.Failure Call.Error -> Worker.Failure
failure (Process.Exit status) = Worker.WorkerExit status
failure (Process.Rejected problem) = Worker.InvalidOutput problem
failure (Process.Protocol problem) = Worker.ProtocolFailure problem

transport :: Worker.Failure -> Process.Failure Call.Error
transport (Worker.WorkerExit status) = Process.Exit status
transport (Worker.InvalidOutput problem) = Process.Rejected problem
transport (Worker.ProtocolFailure problem) = Process.Protocol problem
