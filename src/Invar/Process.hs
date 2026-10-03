{-# LANGUAGE OverloadedStrings #-}

module Invar.Process (Command (..), Launch (..), Exchange (..), Failure (..), run, conversation, batch) where

import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Invar.Process.Internal (Exchange (..), Failure (..), Launch (..), Pipes, Session (..), exited, finish, next, reject, response, send, stage, withLaunch)
import Invar.Transcript (Transcript (..))
import System.Exit (ExitCode (..))

data Command = Command {executable :: FilePath, arguments :: [String], environment :: [(String, String)], input :: ByteString, output :: Transcript}

batch :: Launch -> [Exchange problem] -> IO (Either (Failure problem) [ByteString])
batch _ [] = pure (Right [])
batch launch exchanges = withLaunch launch $ \handles -> exchangeAll (record (transcript launch)) handles exchanges []

exchangeAll :: (ByteString -> IO ()) -> Pipes -> [Exchange problem] -> [ByteString] -> IO (Either (Failure problem) [ByteString])
exchangeAll report handles [] collected = fmap (reverse collected <$) (finish (Session handles (const (pure (Right ""))) report))
exchangeAll report handles (exchange : remaining) collected = do
    send handles (message exchange)
    let session = Session handles (permission exchange) report
    received <- response session exchange ([], False)
    case received of
        Left problem -> pure (Left problem)
        Right output -> exchangeAll report handles remaining (output : collected)

run :: Command -> (ByteString -> IO (Either problem ByteString)) -> IO (Either (Failure problem) ByteString)
run command approve = conversation command approve Nothing

conversation :: Command -> (ByteString -> IO (Either problem ByteString)) -> Maybe (ByteString -> IO (Either problem ByteString)) -> IO (Either (Failure problem) ByteString)
conversation command approve replying = withLaunch launch $ \handles -> do
    send handles (input command)
    consume (Session handles approve (record (output command)), replying) [] False
  where
    launch = Launch (executable command) (arguments command) (environment command) (output command)

consume :: (Session problem, Maybe (ByteString -> IO (Either problem ByteString))) -> [ByteString] -> Bool -> IO (Either (Failure problem) ByteString)
consume context@(session, _) collected granted = do
    received <- next session
    case received of
        Nothing -> complete (pipes session) collected granted
        Just value -> receive context (collected, granted) value

receive :: (Session problem, Maybe (ByteString -> IO (Either problem ByteString))) -> ([ByteString], Bool) -> ByteString -> IO (Either (Failure problem) ByteString)
receive context@(session, replying) (collected, granted) line = do
    let observed = line : collected
    case stage line of
        Left problem -> reject session (Protocol problem)
        Right "consumed" | granted -> reject session (Protocol "Duplicate consumption request")
        Right "consumed" -> authorize context observed
        Right "current" | not granted -> reject session (Protocol "A learner step arrived without approved consumption")
        Right "current" -> case replying of
            Nothing -> reject session (Protocol "This worker does not answer learner steps")
            Just answering -> do
                replied <- answering (Bytes.unlines (reverse observed))
                case replied of
                    Left problem -> reject session (Rejected problem)
                    Right encoded -> send (pipes session) encoded >> consume context observed granted
        Right _ -> consume context observed granted

authorize :: (Session problem, Maybe (ByteString -> IO (Either problem ByteString))) -> [ByteString] -> IO (Either (Failure problem) ByteString)
authorize context@(session, _) collected = do
    permitted <- review session (Bytes.unlines (reverse collected))
    case permitted of
        Left problem -> reject session (Rejected problem)
        Right permission -> send (pipes session) permission >> consume context collected True

complete :: Pipes -> [ByteString] -> Bool -> IO (Either (Failure problem) ByteString)
complete handles collected granted = do
    status <- exited handles
    pure $ case status of
        ExitSuccess | granted -> Right (Bytes.unlines (reverse collected))
        ExitSuccess -> Left (Protocol "Worker exited without approved consumption")
        _ -> Left (Exit status)
