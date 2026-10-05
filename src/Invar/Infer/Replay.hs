module Invar.Infer.Replay (standalone, declared) where

import Control.Monad (foldM)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Invar.Infer.Session qualified as Session
import Invar.Infer.Trajectory (Trajectory)
import Invar.Transcript qualified as Transcript
import System.Exit (ExitCode (..))

standalone :: Session.Protocol -> Session.Declaration -> ExitCode -> ByteString -> Either Session.Error [Trajectory]
standalone protocol declaration status encoded = do
    (_, products) <- foldM advance (Session.start protocol, []) (Session.Dispatched declaration : records encoded ++ [Session.Ended (Transcript.Exited status Transcript.Complete)])
    pure (concat [trajectories | Session.Admitted trajectories <- products])
  where
    advance (current, produced) supplied = fmap (produced ++) <$> Session.step current supplied

declared :: Int -> ExitCode
declared 0 = ExitSuccess
declared status = ExitFailure status

records :: ByteString -> [Session.Input]
records encoded = case reverse (Bytes.split '\n' encoded) of
    remainder : complete -> map Session.Line (reverse complete) ++ [Session.Fragment remainder | not (Bytes.null remainder)]
    [] -> []
