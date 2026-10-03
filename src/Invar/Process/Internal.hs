{-# LANGUAGE OverloadedStrings #-}

module Invar.Process.Internal (Launch (..), Exchange (..), Failure (..), Session (..), Pipes, withLaunch, send, line, next, exited, response, reject, finish, stage) where

import Control.Exception (catch, finally, throwIO)
import Control.Monad (unless)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import Data.Maybe (fromMaybe)
import Data.Text qualified as Text
import GHC.IO.Exception (IOErrorType (ResourceVanished), IOException (ioe_type))
import Invar.Json qualified as Json
import Invar.Transcript (Outcome (..), Transcript (..))
import System.Environment (getEnvironment)
import System.Exit (ExitCode (..))
import System.IO (Handle, hClose, hFlush, hIsEOF)
import System.Process

data Pipes = Pipes Handle Handle ProcessHandle (IORef Bool)

data Launch = Launch {program :: FilePath, launchArguments :: [String], overlay :: [(String, String)], transcript :: Transcript}
data Failure problem = Exit ExitCode | Rejected problem | Protocol String
    deriving (Eq, Show)
data Session problem = Session {pipes :: Pipes, review :: ByteString -> IO (Either problem ByteString), emit :: ByteString -> IO ()}
data Exchange problem = Exchange {message :: ByteString, permission :: ByteString -> IO (Either problem ByteString), completion :: ByteString -> IO (Either problem ()), answer :: Maybe (ByteString -> IO (Either problem ByteString))}

withLaunch :: Launch -> (Pipes -> IO value) -> IO value
withLaunch launch action = do
    prepared <- environmentFor (overlay launch)
    outcome <- newIORef Nothing
    let configured = (proc (program launch) (launchArguments launch)) {std_in = CreatePipe, std_out = CreatePipe, std_err = Inherit, env = prepared}
        observed child stopped = do
            halted <- readIORef stopped
            code <- getProcessExitCode child
            writeIORef outcome (Just (if halted then Stopped else maybe Stopped Exited code))
        run = withCreateProcess configured $ \incoming outgoing _ child -> case (incoming, outgoing) of
            (Just writer, Just reader) -> do
                stopped <- newIORef False
                action (Pipes writer reader child stopped) `finally` observed child stopped
            _ -> ioError (userError "Worker protocol pipes were not created")
        report = readIORef outcome >>= finished (transcript launch) . fromMaybe (Unlaunched "The worker process could not be started")
    run `finally` report

environmentFor :: [(String, String)] -> IO (Maybe [(String, String)])
environmentFor [] = pure Nothing
environmentFor added = do
    inherited <- getEnvironment
    pure (Just ([entry | entry@(name, _) <- inherited, name `notElem` map fst added] ++ added))

send :: Pipes -> ByteString -> IO ()
send (Pipes writer _ _ _) value = vanished (Bytes.hPutStrLn writer value >> hFlush writer)

vanished :: IO () -> IO ()
vanished action = action `catch` \failure -> unless (ioe_type failure == ResourceVanished) (throwIO failure)

line :: Session problem -> IO (Either (Failure problem) ByteString)
line session = do
    received <- next session
    case received of
        Nothing -> do
            status <- exited (pipes session)
            pure $ case status of
                ExitSuccess -> Left (Protocol "Worker exited before a complete response")
                _ -> Left (Exit status)
        Just value -> pure (Right value)

next :: Session problem -> IO (Maybe ByteString)
next session = do
    let Pipes _ reader _ _ = pipes session
    ended <- hIsEOF reader
    if ended
        then pure Nothing
        else do
            received <- Bytes.hGetLine reader
            emit session received
            pure (Just received)

exited :: Pipes -> IO ExitCode
exited (Pipes _ _ child _) = waitForProcess child

response :: Session problem -> Exchange problem -> ([ByteString], Bool) -> IO (Either (Failure problem) ByteString)
response session exchange state = do
    received <- line session
    case received of
        Left problem -> pure (Left problem)
        Right receivedLine -> responseLine session (exchange, state) receivedLine

responseLine :: Session problem -> (Exchange problem, ([ByteString], Bool)) -> ByteString -> IO (Either (Failure problem) ByteString)
responseLine session (exchange, (collected, granted)) received = do
    let observed = received : collected
        output = Bytes.unlines (reverse observed)
    case stage received of
        Left problem -> reject session (Protocol problem)
        Right "consumed" | granted -> reject session (Protocol "Duplicate consumption request")
        Right "consumed" -> permitResponse session (exchange, observed) output
        Right "current" | not granted -> reject session (Protocol "A learner step arrived without approved consumption")
        Right "current" -> case answer exchange of
            Nothing -> reject session (Protocol "This exchange does not answer learner steps")
            Just replying -> do
                replied <- replying output
                case replied of
                    Left problem -> reject session (Rejected problem)
                    Right encoded -> send (pipes session) encoded >> response session exchange (observed, granted)
        Right "result" | not granted -> reject session (Protocol "Batch result arrived without approved consumption")
        Right "result" -> do
            completed <- completion exchange output
            case completed of
                Left problem -> reject session (Rejected problem)
                Right () -> pure (Right output)
        Right _ -> response session exchange (observed, granted)

permitResponse :: Session problem -> (Exchange problem, [ByteString]) -> ByteString -> IO (Either (Failure problem) ByteString)
permitResponse session (exchange, observed) output = do
    permitted <- permission exchange output
    case permitted of
        Left problem -> reject session (Rejected problem)
        Right allowed -> send (pipes session) allowed >> response session exchange (observed, True)

reject :: Session problem -> Failure problem -> IO (Either (Failure problem) value)
reject session problem = do
    let Pipes _ _ child stopped = pipes session
    writeIORef stopped True
    terminateProcess child
    _ <- waitForProcess child
    pure (Left problem)

finish :: Session problem -> IO (Either (Failure problem) ())
finish session = do
    let Pipes writer reader child _ = pipes session
    vanished (hClose writer)
    ended <- hIsEOF reader
    if ended
        then do
            status <- waitForProcess child
            pure $ case status of
                ExitSuccess -> Right ()
                _ -> Left (Exit status)
        else reject session (Protocol "Output follows the final batch response")

stage :: ByteString -> Either String String
stage encoded = Text.unpack <$> Json.textField "stage" encoded
