module Invar.Infer.Replay (Logged, Pending, standalone, session, delimit, group, declared, source, trajectory) where

import Control.Monad (foldM)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Invar.Artifact qualified as Artifact
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Session qualified as Session
import Invar.Infer.Trajectory (Trajectory)
import Invar.Resident.Owner qualified as Owner
import Invar.Transcript qualified as Transcript
import System.Exit (ExitCode (..))

data Logged = Logged String Trajectory

data Pending = Pending Session.Session ByteString

standalone :: Session.Protocol -> Session.Declaration -> ExitCode -> ByteString -> Either Session.Error [Logged]
standalone protocol declaration status encoded = do
    (_, products) <- foldM advance (Session.start protocol, []) (Session.Dispatched declaration : records encoded ++ [Session.Ended (Transcript.Exited status Transcript.Complete)])
    pure (sealed encoded products)
  where
    advance (current, produced) supplied = fmap (produced ++) <$> Session.step current supplied

session :: Session.Protocol -> Session.Declaration -> [Framing.Frame] -> Either Session.Error (Pending, [Framing.Frame], [Framing.Frame])
session protocol declaration frames = do
    (opened, _) <- Session.step (Session.start protocol) (Session.Dispatched declaration)
    consume opened [] frames
  where
    consume current taken remaining = case remaining of
        [] -> Left (Session.Protocol "Session segment ends before its final response")
        frame : rest -> do
            (following, products) <- Session.step current (Session.Line (Framing.raw frame))
            let consumed = taken ++ [frame]
            if any closing products then pure (Pending following (Framing.encode consumed), consumed, rest) else consume following consumed rest
    closing Session.Close = True
    closing _ = False

delimit :: Pending -> Either Session.Error [Logged]
delimit (Pending current encoded) = sealed encoded . snd <$> Session.step current Session.Delimited

group :: (Session.Session, Owner.State) -> Session.Declaration -> [Framing.Frame] -> Either Session.Error ((Session.Session, Owner.State), [Logged], [Framing.Frame], [Framing.Frame])
group (current, physical) declaration frames = do
    (opened, _) <- Session.step current (Session.Hosted physical declaration)
    consume opened [] frames
  where
    consume active taken remaining = case remaining of
        [] -> Left (Session.Protocol "Resident group ends before its release acknowledgement")
        frame : rest -> do
            (following, products) <- Session.step active (Session.Line (Framing.raw frame))
            let consumed = taken ++ [frame]
            case [released | Session.Owned released <- products] of
                [released] -> pure ((following, released), sealed (Framing.encode consumed) products, consumed, rest)
                _ -> consume following consumed rest

sealed :: ByteString -> [Session.Product] -> [Logged]
sealed encoded products = [Logged identity admitted | Session.Admitted trajectories <- products, admitted <- trajectories]
  where
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
