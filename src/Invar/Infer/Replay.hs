module Invar.Infer.Replay (Logged, standalone, declared, source, trajectory) where

import Control.Monad (foldM)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Invar.Artifact qualified as Artifact
import Invar.Infer.Session qualified as Session
import Invar.Infer.Trajectory (Trajectory)
import Invar.Transcript qualified as Transcript
import System.Exit (ExitCode (..))

data Logged = Logged String Trajectory

standalone :: Session.Protocol -> Session.Declaration -> ExitCode -> ByteString -> Either Session.Error [Logged]
standalone protocol declaration status encoded = do
    (_, products) <- foldM advance (Session.start protocol, []) (Session.Dispatched declaration : records encoded ++ [Session.Ended (Transcript.Exited status Transcript.Complete)])
    pure [Logged identity admitted | Session.Admitted trajectories <- products, admitted <- trajectories]
  where
    advance (current, produced) supplied = fmap (produced ++) <$> Session.step current supplied
    identity = Artifact.hex (SHA256.hash encoded)

source :: Logged -> String
source (Logged identity _) = identity

trajectory :: Logged -> Trajectory
trajectory (Logged _ admitted) = admitted

declared :: Int -> ExitCode
declared 0 = ExitSuccess
declared status = ExitFailure status

records :: ByteString -> [Session.Input]
records encoded = case reverse (Bytes.split '\n' encoded) of
    remainder : complete -> map Session.Line (reverse complete) ++ [Session.Fragment remainder | not (Bytes.null remainder)]
    [] -> []
