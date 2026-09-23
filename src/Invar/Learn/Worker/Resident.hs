{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RoleAnnotations #-}

module Invar.Learn.Worker.Resident (Options (..), Paths (..), Resident, Receipt, withResident, withBorrowed, run, report, plan, staged, loaded, acknowledgement) where

import Control.Exception (bracket, mask_)
import Control.Monad (unless, void)
import Data.Aeson (encode, object, (.=))
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Lazy qualified as Lazy
import Data.IORef (IORef, modifyIORef', newIORef, readIORef, writeIORef)
import Invar.Learn qualified as L
import Invar.Learn.Framing qualified as Framing
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Wire qualified as Wire
import Invar.Learn.Worker.Internal qualified as W
import Invar.Process qualified as Process
import Invar.Process.Resident qualified as Transport
import Invar.Resident qualified as Boundary
import Invar.Spec.Load qualified as Load
import Numeric.Natural (Natural)

data Options = Options {worker :: W.Worker, owner :: Natural, echo :: ByteString -> IO ()}
data Paths = Paths {checkpoint :: FilePath, output :: FilePath}
data State = State Load.Registry Natural

type role Receipt nominal
data Receipt scope = Receipt (L.Plan scope) P.Result FilePath Load.Fact ByteString

data Progress scope = Awaiting | Consumed P.Permit | Completed P.Result Load.Fact Boundary.Release | Released (Receipt scope)

type role Resident nominal
data Resident scope = Resident (Transport.Resident scope) (IORef State) Boundary.Owner

withResident :: Options -> (forall scope. Resident scope -> IO (Either W.Failure value)) -> IO (Either W.Failure value)
withResident options action = bracket (newIORef (State Load.empty 0)) retireOwner $ \state -> do
    let selected = worker options
        arguments = [W.script selected, "--cache=" ++ W.cache selected, "--reference=" ++ W.reference selected, "--session=" ++ show (owner options)]
        launch = Process.Launch (W.executable selected) arguments [] (echo options)
        identity = Boundary.Owner Boundary.Learning (owner options)
        closing = Transport.Handshake (Boundary.close identity) (close identity state)
    first failure <$> Transport.withResident launch (const closing) (\process -> first Process.Rejected <$> action (Resident process state identity))
  where
    retireOwner state = modifyIORef' state (\(State registry groups) -> State (Load.close registry) groups)
    close identity state encoded = do
        State registry groups <- readIORef state
        pure $ first W.ProtocolFailure $ do
            unless (null (Load.active registry)) (Left "Learner closes with active invocation loads")
            void (Boundary.closed identity groups encoded)

withBorrowed :: Transport.Resident scope -> Boundary.Owner -> (Resident scope -> IO value) -> IO value
withBorrowed process identity action = bracket (newIORef (State Load.empty 0)) retire $ \state -> do
    returned <- action (Resident process state identity)
    State registry _ <- readIORef state
    unless (null (Load.active registry)) (ioError (userError "Borrowed learner role exits with active invocation loads"))
    pure returned
  where
    retire state = modifyIORef' state (\(State registry count) -> State (Load.close registry) count)

run :: Resident owner -> Paths -> W.Call scope -> IO (Either W.Failure (Receipt scope))
run (Resident process state identity) paths call = do
    progress <- newIORef Awaiting
    let message = Lazy.toStrict (encode (object ["format" .= ("invar-learning-resident-v1" :: String), "checkpoint" .= checkpoint paths, "output" .= output paths, "call" .= W.input call]))
        exchange = Process.Exchange message (authorize process (state, progress) call) (complete (identity, output paths) progress)
        transaction = Transport.Transaction exchange (release (state, progress) (paths, call))
    returned <- Transport.exchange process transaction
    case first failure returned of
        Left problem -> pure (Left problem)
        Right _ -> do
            finished <- readIORef progress
            pure $ case finished of
                Released value -> Right value
                _ -> Left (W.ProtocolFailure "Resident update has no checked release acknowledgement")

authorize :: Transport.Resident owner -> (IORef State, IORef (Progress scope)) -> W.Call scope -> ByteString -> IO (Either W.Failure ByteString)
authorize process (state, progress) call@(W.Call planned binding runtime _) encoded = mask_ $ do
    current <- readIORef progress
    State registry groups <- readIORef state
    physical <- Transport.groups process
    case current of
        Awaiting -> case admit registry physical of
            Left problem -> pure (Left problem)
            Right (updated, permit) -> do
                writeIORef state (State updated groups)
                writeIORef progress (Consumed permit)
                pure (Right (W.permission call))
        _ -> pure (Left (W.ProtocolFailure "Resident update already holds a consumption permit"))
  where
    admit registry groups = do
        request <- first W.Lowering (Wire.lower (L.emission planned))
        first W.ProtocolFailure (Framing.readiness (groups == 0) request encoded)
        first W.InvalidOutput (P.authorizeResident registry (binding, runtime) encoded)

complete :: (Boundary.Owner, FilePath) -> IORef (Progress scope) -> ByteString -> IO (Either W.Failure ())
complete (identity, directory) progress encoded = do
    current <- readIORef progress
    case current of
        Consumed permit -> case observe permit of
            Left problem -> pure (Left problem)
            Right (result, fact, prepared) -> do
                verified <- W.verifyArtifacts directory result
                case verified of
                    Left problem -> pure (Left problem)
                    Right () -> writeIORef progress (Completed result fact prepared) >> pure (Right ())
        _ -> pure (Left (W.ProtocolFailure "Resident update has no accepted consumption permit"))
  where
    observe permit = do
        result <- first W.InvalidOutput (P.observe permit encoded)
        first W.ProtocolFailure (Framing.completion (P.request result) encoded)
        let fact = P.loadedFact permit
        prepared <- first W.ProtocolFailure (Boundary.prepare identity [fact] encoded)
        pure (result, fact, prepared)

release :: (IORef State, IORef (Progress scope)) -> (Paths, W.Call scope) -> ByteString -> IO (Either W.Failure (Transport.Handshake W.Failure))
release (state, progress) (paths, W.Call planned _ _ _) _ = do
    current <- readIORef progress
    pure $ case current of
        Completed result fact prepared -> Right (Transport.Handshake (Boundary.request prepared) (acknowledge prepared result fact))
        _ -> Left (W.ProtocolFailure "Learner release precedes a checked update and its artifacts")
  where
    acknowledge prepared result fact encoded = mask_ $ do
        State registry groups <- readIORef state
        case first W.ProtocolFailure (Boundary.retire prepared registry encoded) of
            Left problem -> pure (Left problem)
            Right updated -> do
                writeIORef state (State updated (groups + 1))
                writeIORef progress (Released (Receipt planned result (output paths) fact encoded))
                pure (Right ())

report :: Receipt scope -> P.Result
report (Receipt _ result _ _ _) = result

plan :: Receipt scope -> L.Plan scope
plan (Receipt planned _ _ _ _) = planned

staged :: Receipt scope -> FilePath
staged (Receipt _ _ directory _ _) = directory

loaded :: Receipt scope -> Load.Fact
loaded (Receipt _ _ _ fact _) = fact

acknowledgement :: Receipt scope -> ByteString
acknowledgement (Receipt _ _ _ _ encoded) = encoded

failure :: Process.Failure W.Failure -> W.Failure
failure (Process.Exit status) = W.WorkerExit status
failure (Process.Rejected problem) = problem
failure (Process.Protocol problem) = W.ProtocolFailure problem
