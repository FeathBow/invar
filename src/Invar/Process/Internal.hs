{-# LANGUAGE OverloadedStrings #-}

module Invar.Process.Internal (Launch (..), Exchange (..), Failure (..), Session (..), Pipes, Received (..), withLaunch, send, shut, line, next, receive, exited, halt, response, reject, finish, stage) where

import Control.Exception (catch, finally, mask_, onException, throwIO, tryJust)
import Control.Monad (guard, unless)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.IORef (IORef, modifyIORef', newIORef, readIORef, writeIORef)
import Data.Maybe (fromMaybe)
import Data.Text qualified as Text
import GHC.IO.Exception (IOErrorType (ResourceVanished), IOException (ioe_type))
import Invar.Json qualified as Json
import Invar.Transcript (Outcome (..), Output (..), Transcript (..))
import System.Environment (getEnvironment)
import System.Exit (ExitCode (..))
import System.IO (Handle, hClose, hFlush, hIsEOF, hReady, hSetBinaryMode)
import System.IO.Error (isEOFError)
import System.Process

data Pipes = Pipes Handle Handle ProcessHandle Transcript (IORef Reading)

data Reading = Reading {pending :: [ByteString], ended :: Bool, stopped :: Bool, broken :: Bool}

data Received = Line ByteString | Fragment ByteString | Exhausted

data Launch = Launch {program :: FilePath, launchArguments :: [String], overlay :: [(String, String)], transcript :: Transcript}
data Failure problem = Exit ExitCode | Rejected problem | Protocol String
    deriving (Eq, Show)
data Session problem = Session {pipes :: Pipes, review :: ByteString -> IO (Either problem ByteString)}
data Exchange problem = Exchange {message :: ByteString, permission :: ByteString -> IO (Either problem ByteString), completion :: ByteString -> IO (Either problem ()), answer :: Maybe (ByteString -> IO (Either problem ByteString))}

withLaunch :: Launch -> (Pipes -> IO value) -> IO value
withLaunch launch action = do
    prepared <- environmentFor (overlay launch)
    outcome <- newIORef Nothing
    let configured = (proc (program launch) (launchArguments launch)) {std_in = CreatePipe, std_out = CreatePipe, std_err = Inherit, env = prepared}
        observed child state = do
            current <- readIORef state
            code <- getProcessExitCode child
            let read' = if ended current && not (broken current) then Complete else Cut
            writeIORef outcome (Just (if stopped current then Stopped read' else maybe (Stopped read') (`Exited` read') code))
        run = withCreateProcess configured $ \incoming outgoing _ child -> case (incoming, outgoing) of
            (Just writer, Just reader) -> do
                state <- newIORef (Reading [] False False False)
                (hSetBinaryMode writer True >> hSetBinaryMode reader True >> action (Pipes writer reader child (transcript launch) state)) `finally` observed child state
            _ -> ioError (userError "Worker protocol pipes were not created")
        report = readIORef outcome >>= finished (transcript launch) . fromMaybe (Unlaunched "The worker process could not be started")
    run `finally` report

environmentFor :: [(String, String)] -> IO (Maybe [(String, String)])
environmentFor [] = pure Nothing
environmentFor added = do
    inherited <- getEnvironment
    pure (Just ([entry | entry@(name, _) <- inherited, name `notElem` map fst added] ++ added))

send :: Pipes -> ByteString -> IO ()
send (Pipes writer _ _ _ _) value = vanished (Bytes.hPutStrLn writer value >> hFlush writer)

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
    received <- receive (pipes session)
    pure $ case received of
        Line value -> Just value
        Fragment value -> Just value
        Exhausted -> Nothing

receive :: Pipes -> IO Received
receive handles@(Pipes _ reader _ recorded state) = do
    held <- pending <$> readIORef state
    case held of
        newest : older | Just position <- Bytes.elemIndex '\n' newest -> do
            let received = Bytes.concat (reverse (Bytes.take position newest : older))
                rest = Bytes.drop (position + 1) newest
            settle state (record recorded received) (\current -> current {pending = [rest | not (Bytes.null rest)]})
            pure (Line received)
        _ -> do
            count <- mask_ (Bytes.hGetSome reader 65536 >>= kept state)
            if count == 0
                then do
                    let received = Bytes.concat (reverse held)
                    settle state (unless (Bytes.null received) (partial recorded received)) (\current -> current {pending = [], ended = True})
                    pure (if Bytes.null received then Exhausted else Fragment received)
                else receive handles

shut :: Pipes -> IO ()
shut (Pipes writer _ _ _ _) = vanished (hClose writer)

kept :: IORef Reading -> ByteString -> IO Int
kept state chunk = do
    unless (Bytes.null chunk) (modifyIORef' state (\current -> current {pending = chunk : pending current}))
    pure (Bytes.length chunk)

settle :: IORef Reading -> IO () -> (Reading -> Reading) -> IO ()
settle state write change = mask_ $ do
    current <- readIORef state
    unless (broken current) (write `onException` modifyIORef' state (\reading -> reading {broken = True}))
    modifyIORef' state change

exited :: Pipes -> IO ExitCode
exited (Pipes _ _ child _ _) = waitForProcess child

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
reject session problem = halt (pipes session) >> pure (Left problem)

halt :: Pipes -> IO ()
halt handles@(Pipes _ _ child _ state) = do
    modifyIORef' state (\current -> current {stopped = True})
    terminateProcess child
    _ <- waitForProcess child
    drain handles

drain :: Pipes -> IO ()
drain (Pipes _ reader _ recorded state) = do
    held <- pending <$> readIORef state
    exhausted <- gather (remainder - sum (map Bytes.length held))
    modifyIORef' state (\current -> current {pending = filter (not . Bytes.null) [Bytes.concat (reverse (pending current))]})
    flush exhausted
  where
    gather remaining = do
        ready <- tryJust (guard . isEOFError) (hReady reader)
        case ready of
            Left () -> pure True
            Right True | remaining > 0 -> do
                count <- mask_ (Bytes.hGetNonBlocking reader (min remaining 65536) >>= kept state)
                if count == 0 then pure False else gather (remaining - count)
            Right _ -> pure False
    flush exhausted = do
        held <- pending <$> readIORef state
        case held of
            [bytes] | Just position <- Bytes.elemIndex '\n' bytes -> do
                let rest = Bytes.drop (position + 1) bytes
                settle state (record recorded (Bytes.take position bytes)) (\current -> current {pending = [rest | not (Bytes.null rest)]})
                flush exhausted
            [bytes] -> settle state (partial recorded bytes) (\current -> current {pending = [], ended = exhausted})
            _ -> modifyIORef' state (\current -> current {ended = exhausted})

remainder :: Int
remainder = 1048576

finish :: Session problem -> IO (Either (Failure problem) ())
finish session = do
    let Pipes writer reader child _ state = pipes session
    vanished (hClose writer)
    held <- pending <$> readIORef state
    exhausted <- if null held then hIsEOF reader else pure False
    if exhausted
        then do
            modifyIORef' state (\current -> current {ended = True})
            status <- waitForProcess child
            pure $ case status of
                ExitSuccess -> Right ()
                _ -> Left (Exit status)
        else reject session (Protocol "Output follows the final batch response")

stage :: ByteString -> Either String String
stage encoded = Text.unpack <$> Json.textField "stage" encoded
