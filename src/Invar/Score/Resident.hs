{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RoleAnnotations #-}

module Invar.Score.Resident (Resident, withResident, run, readiness) where

import Control.Exception (bracket, mask_)
import Control.Monad (unless)
import Data.Aeson (Value (..))
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.IORef (IORef, modifyIORef', newIORef, readIORef, writeIORef)
import Invar.Infer.Framing qualified as Framing
import Invar.Process qualified as Process
import Invar.Process.Resident qualified as Transport
import Invar.Resident qualified as Boundary
import Invar.Resident.Owner qualified as Owner
import Invar.Score qualified as Score
import Invar.Score.Worker (Failure (..))
import Invar.Spec.Load qualified as Load
import Invar.Transcript qualified as Transcript
import Invar.Worker qualified as Worker
import Numeric.Natural (Natural)

data Progress = Awaiting | Consumed Score.Permit | Completed Score.Report Boundary.Release | Released Score.Report

type role Resident nominal
data Resident scope = Resident (Transport.Resident scope) (IORef Load.Registry) Boundary.Owner

withResident :: Worker.Worker -> Natural -> Transcript.Transcript -> (forall scope. Resident scope -> IO (Either Failure value)) -> IO (Either Failure value)
withResident worker session transcript action = bracket (newIORef Load.empty) (`modifyIORef'` Load.close) $ \state -> do
    let arguments = [Worker.script worker, "--cache=" ++ Worker.cache worker, "--adapter=" ++ Worker.adapter worker, "--session=" ++ show session] ++ maybe [] (\path -> ["--config=" ++ path]) (Worker.configuration worker)
        launch = Process.Launch (Worker.executable worker) arguments (Worker.environment worker) transcript
        identity = Boundary.Owner Boundary.Teacher session
        ready _ = do
            registry <- readIORef state
            pure (if null (Load.active registry) then Right () else Left (ProtocolFailure "The teacher closes with active invocation loads"))
    first failure <$> Transport.withResident launch identity ready (\process -> first Process.Rejected <$> action (Resident process state identity))

run :: Resident scope -> Score.Call -> IO (Either Failure Score.Report)
run (Resident process state identity) call = do
    progress <- newIORef Awaiting
    let exchange = Process.Exchange (Score.input call) (authorize process (state, progress) call) (complete identity progress) Nothing
    returned <- Transport.exchange process (Transport.Transaction exchange (release (state, progress) call))
    case first failure returned of
        Left problem -> pure (Left problem)
        Right _ -> do
            finished <- readIORef progress
            pure $ case finished of
                Released scored -> Right scored
                _ -> Left (ProtocolFailure "The teacher score has no checked release acknowledgement")

authorize :: Transport.Resident scope -> (IORef Load.Registry, IORef Progress) -> Score.Call -> ByteString -> IO (Either Failure ByteString)
authorize process (state, progress) call encoded = mask_ $ do
    current <- readIORef progress
    registry <- readIORef state
    physical <- Transport.initial process
    case current of
        Awaiting -> case readiness physical encoded >> first InvalidOutput (Score.authorize registry call encoded) of
            Left problem -> pure (Left problem)
            Right (updated, permit) -> do
                writeIORef state updated
                writeIORef progress (Consumed permit)
                pure (Right (Score.permission permit))
        _ -> pure (Left (ProtocolFailure "The teacher score already holds a consumption permit"))

readiness :: Bool -> ByteString -> Either Failure ()
readiness initial encoded = do
    records <- first ProtocolFailure (Framing.decode encoded)
    let prefix = takeWhile ((`elem` map (Just . String) ["loading", "profile", "load"]) . Framing.stageName) records
        expected = if initial then [["load"], ["loading", "profile", "load"]] else [[]]
    unless (map Framing.stageName prefix `elem` map (map (Just . String)) expected) (Left (ProtocolFailure "Expected one initial teacher load and none for later scores"))

complete :: Boundary.Owner -> IORef Progress -> ByteString -> IO (Either Failure ())
complete identity progress encoded = do
    current <- readIORef progress
    case current of
        Consumed permit -> case observed permit of
            Left problem -> pure (Left problem)
            Right (scored, prepared) -> writeIORef progress (Completed scored prepared) >> pure (Right ())
        _ -> pure (Left (ProtocolFailure "The teacher score has no accepted consumption permit"))
  where
    observed permit = do
        scored <- first InvalidOutput (Score.observe permit encoded)
        prepared <- first ProtocolFailure (Boundary.prepare identity [Score.loadFact scored] encoded)
        pure (scored, prepared)

release :: (IORef Load.Registry, IORef Progress) -> Score.Call -> ByteString -> IO (Either Failure (Transport.Handshake Failure Owner.Released))
release (state, progress) call produced = do
    current <- readIORef progress
    pure $ case current of
        Completed scored prepared -> Right (Transport.Handshake (Boundary.request prepared) (acknowledge prepared scored))
        _ -> Left (ProtocolFailure "The teacher release precedes a checked score")
  where
    acknowledge prepared scored encoded = mask_ $ do
        registry <- readIORef state
        case first ProtocolFailure (retired prepared registry encoded) of
            Left problem -> pure (Left problem)
            Right (updated, group) -> do
                writeIORef state updated
                writeIORef progress (Released scored)
                pure (Right group)
    retired prepared registry encoded = do
        updated <- Boundary.retire prepared registry encoded
        frames <- Framing.decode produced
        let loadedRecords = [Framing.fields frame | frame <- frames, Framing.stageName frame == Just (String "loaded_adapter")]
        pure (updated, Owner.Released (map Framing.raw frames) [Score.binding call] loadedRecords encoded)

failure :: Process.Failure Failure -> Failure
failure (Process.Exit status) = WorkerExit status
failure (Process.Rejected problem) = problem
failure (Process.Protocol problem) = ProtocolFailure problem
