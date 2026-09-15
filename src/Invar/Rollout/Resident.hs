module Invar.Rollout.Resident (Pool, withPool, borrowed, matches, sessions, flush) where

import Control.Exception (finally)
import Control.Monad (join)
import Data.ByteString (ByteString)
import Data.IORef (atomicModifyIORef', modifyIORef', newIORef)
import Invar.Infer.Invocation qualified as Call
import Invar.Worker qualified as Worker
import Invar.Worker.Resident qualified as Resident

type Configuration = (Worker.Worker, [[(String, String)]])
type Session = FilePath -> [Call.Call] -> IO (Either Worker.Failure [Resident.Receipt])

data Pool = Pool Configuration [Session] (IO ())

borrowed :: Configuration -> Resident.Resident scope -> Pool
borrowed configuration resident = Pool configuration [Resident.run resident] (pure ())

withPool :: Configuration -> (ByteString -> IO ()) -> (Pool -> IO value) -> IO (Either Worker.Failure value)
withPool configuration@(worker, overlays) echo action = open (zip [0 ..] overlays) [] []
  where
    open [] workers drains = Right <$> action (Pool configuration (reverse workers) (sequence_ (reverse drains)))
    open ((index, overlay) : remaining) workers drains = do
        buffer <- newIORef []
        let drain = atomicModifyIORef' buffer (\emitted -> ([], reverse emitted)) >>= mapM_ echo
            selected = Resident.Options (worker {Worker.environment = overlay}) index (\line -> modifyIORef' buffer (line :))
        returned <- Resident.withResident selected (\resident -> Right <$> open remaining (Resident.run resident : workers) (drain : drains)) `finally` drain
        pure (join returned)

matches :: Pool -> Configuration -> Bool
matches (Pool (original, overlays) _ _) (requested, environments) =
    launch original == launch requested && overlays == environments
  where
    launch worker = (Worker.executable worker, Worker.script worker, Worker.cache worker, Worker.configuration worker, Worker.qualificationFile worker)

sessions :: Pool -> [Session]
sessions (Pool _ workers _) = workers

flush :: Pool -> IO ()
flush (Pool _ _ drain) = drain
