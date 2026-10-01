{-# LANGUAGE OverloadedStrings #-}

module Invar.Process (Command (..), Launch (..), Exchange (..), Failure (..), run, conversation, batch, live) where

import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Invar.Process.Internal (Exchange (..), Failure (..), Launch (..), Pipes, Session (..), finish, reject, response, send, stage, withLaunch)
import System.Exit (ExitCode (..))
import System.IO (hFlush, hIsEOF, stdout)
import System.Process (ProcessHandle, waitForProcess)

data Command = Command {executable :: FilePath, arguments :: [String], environment :: [(String, String)], input :: ByteString}

batch :: Launch -> [Exchange problem] -> IO (Either (Failure problem) [ByteString])
batch _ [] = pure (Right [])
batch launch exchanges = withLaunch launch $ \handles -> exchangeAll (echo launch) handles exchanges []

exchangeAll :: (ByteString -> IO ()) -> Pipes -> [Exchange problem] -> [ByteString] -> IO (Either (Failure problem) [ByteString])
exchangeAll _ handles [] collected = fmap (reverse collected <$) (finish (Session handles (const (pure (Right ""))) (const (pure ()))))
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
    consume (Session handles approve live, replying) [] False
  where
    launch = Launch (executable command) (arguments command) (environment command) live

live :: ByteString -> IO ()
live line = Bytes.hPut stdout (Bytes.snoc line '\n') >> hFlush stdout

consume :: (Session problem, Maybe (ByteString -> IO (Either problem ByteString))) -> [ByteString] -> Bool -> IO (Either (Failure problem) ByteString)
consume context@(session, _) collected granted = do
    let (_, reader, child) = pipes session
    ended <- hIsEOF reader
    if ended
        then complete child collected granted
        else Bytes.hGetLine reader >>= receive context (collected, granted)

receive :: (Session problem, Maybe (ByteString -> IO (Either problem ByteString))) -> ([ByteString], Bool) -> ByteString -> IO (Either (Failure problem) ByteString)
receive context@(session, replying) (collected, granted) line = do
    emit session line
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

complete :: ProcessHandle -> [ByteString] -> Bool -> IO (Either (Failure problem) ByteString)
complete child collected granted = do
    status <- waitForProcess child
    pure $ case status of
        ExitSuccess | granted -> Right (Bytes.unlines (reverse collected))
        ExitSuccess -> Left (Protocol "Worker exited without approved consumption")
        _ -> Left (Exit status)
