module Invar.Rollout.Resident (Pool, withPool, borrowed, matches, sessions, flush) where

import Control.Exception (finally)
import Control.Monad (join)
import Data.IORef (atomicModifyIORef', modifyIORef', newIORef)
import Invar.Infer.Batch qualified as Batch
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Trajectory (Trajectory)
import Invar.Transcript qualified as Transcript
import Invar.Worker qualified as Worker
import Invar.Worker.Resident qualified as Resident
import Numeric.Natural (Natural)

type Configuration = (Worker.Worker, [[(String, String)]])
type Session = FilePath -> Maybe Batch.Reference -> [Call.Call] -> IO (Either Worker.Failure [Trajectory])

data Pool = Pool Configuration [Session] (IO ())

borrowed :: Configuration -> Resident.Resident scope -> Pool
borrowed configuration resident = Pool configuration [Resident.run resident] (pure ())

withPool :: Configuration -> Maybe (Natural -> IO Transcript.Transcript) -> (Pool -> IO value) -> IO (Either Worker.Failure value)
withPool configuration@(worker, overlays) recorded action = open (zip [0 ..] overlays) [] []
  where
    open [] workers drains = Right <$> action (Pool configuration (reverse workers) (sequence_ (reverse drains)))
    open ((index, overlay) : remaining) workers drains = do
        (transcript, drain) <- case recorded of
            Just opened -> (,pure ()) <$> opened index
            Nothing -> do
                buffer <- newIORef []
                pure (Transcript.echoing (\line -> modifyIORef' buffer (line :)), atomicModifyIORef' buffer (\emitted -> ([], reverse emitted)) >>= mapM_ Transcript.live)
        let selected = Resident.Options (worker {Worker.environment = overlay}) index transcript
        returned <- Resident.withResident selected (\resident -> Right <$> open remaining (Resident.run resident : workers) (drain : drains)) `finally` drain
        pure (join returned)

matches :: Pool -> Configuration -> Bool
matches (Pool (original, overlays) _ _) (requested, environments) =
    launch original == launch requested && overlays == environments
  where
    launch worker = (Worker.executable worker, Worker.script worker, Worker.cache worker, Worker.configuration worker)

sessions :: Pool -> [Session]
sessions (Pool _ workers _) = workers

flush :: Pool -> IO ()
flush (Pool _ _ drain) = drain
