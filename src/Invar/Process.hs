{-# LANGUAGE OverloadedStrings #-}

module Invar.Process (Command (..), Launch (..), Exchange (..), Failure (..), run, batch) where

import Data.Aeson (eitherDecodeStrict, (.:))
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import System.Environment (getEnvironment)
import System.Exit (ExitCode (..))
import System.IO (Handle, hClose, hFlush, hIsEOF, stdout)
import System.Process

data Command = Command {executable :: FilePath, arguments :: [String], environment :: [(String, String)], input :: ByteString}

data Launch = Launch {program :: FilePath, launchArguments :: [String], overlay :: [(String, String)], echo :: ByteString -> IO ()}

data Failure problem = Exit ExitCode | Rejected problem | Protocol String
    deriving (Eq, Show)

data Session problem = Session
    { pipes :: (Handle, Handle, ProcessHandle)
    , review :: ByteString -> IO (Either problem ByteString)
    , emit :: ByteString -> IO ()
    }

data Exchange problem = Exchange
    { message :: ByteString
    , permission :: ByteString -> IO (Either problem ByteString)
    , completion :: ByteString -> IO (Either problem ())
    }

batch :: Launch -> [Exchange problem] -> IO (Either (Failure problem) [ByteString])
batch _ [] = pure (Right [])
batch launch exchanges = do
    prepared <- environmentFor (overlay launch)
    let configured = (proc (program launch) (launchArguments launch)) {std_in = CreatePipe, std_out = CreatePipe, std_err = Inherit, env = prepared}
    withCreateProcess configured $ \incoming outgoing _ child -> case (incoming, outgoing) of
        (Just writer, Just reader) -> exchangeAll (echo launch) (writer, reader, child) exchanges []
        _ -> ioError (userError "Worker batch pipes were not created")

environmentFor :: [(String, String)] -> IO (Maybe [(String, String)])
environmentFor [] = pure Nothing
environmentFor added = do
    inherited <- getEnvironment
    pure (Just ([entry | entry@(name, _) <- inherited, name `notElem` map fst added] ++ added))

exchangeAll :: (ByteString -> IO ()) -> (Handle, Handle, ProcessHandle) -> [Exchange problem] -> [ByteString] -> IO (Either (Failure problem) [ByteString])
exchangeAll _ handles@(writer, reader, child) [] collected = do
    hClose writer
    ended <- hIsEOF reader
    if ended
        then do
            status <- waitForProcess child
            pure $ case status of
                ExitSuccess -> Right (reverse collected)
                _ -> Left (Exit status)
        else reject (Session handles (const (pure (Right ""))) (const (pure ()))) (Protocol "Output follows the final batch response")
exchangeAll report handles@(writer, _, _) (exchange : remaining) collected = do
    Bytes.hPutStrLn writer (message exchange)
    hFlush writer
    let session = Session handles (permission exchange) report
    received <- response session exchange ([], False)
    case received of
        Left problem -> pure (Left problem)
        Right output -> exchangeAll report handles remaining (output : collected)

response :: Session problem -> Exchange problem -> ([ByteString], Bool) -> IO (Either (Failure problem) ByteString)
response session exchange state = do
    let (_, reader, child) = pipes session
    ended <- hIsEOF reader
    if ended
        then do
            status <- waitForProcess child
            pure $ case status of
                ExitSuccess -> Left (Protocol "Worker exited before a complete batch response")
                _ -> Left (Exit status)
        else Bytes.hGetLine reader >>= responseLine session (exchange, state)

responseLine :: Session problem -> (Exchange problem, ([ByteString], Bool)) -> ByteString -> IO (Either (Failure problem) ByteString)
responseLine session (exchange, (collected, granted)) line = do
    emit session line
    let observed = line : collected
        output = Bytes.unlines (reverse observed)
    case stage line of
        Left problem -> reject session (Protocol problem)
        Right "consumed" | granted -> reject session (Protocol "Duplicate consumption request")
        Right "consumed" -> permitResponse session (exchange, observed) output
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
        Right allowed -> do
            let (writer, _, _) = pipes session
            Bytes.hPutStrLn writer allowed
            hFlush writer
            response session exchange (observed, True)

run :: Command -> (ByteString -> IO (Either problem ByteString)) -> IO (Either (Failure problem) ByteString)
run command approve = do
    prepared <- environmentFor (environment command)
    let configured = (proc (executable command) (arguments command)) {std_in = CreatePipe, std_out = CreatePipe, std_err = Inherit, env = prepared}
    withCreateProcess configured $ \incoming outgoing _ child -> case (incoming, outgoing) of
        (Just writer, Just reader) -> do
            Bytes.hPutStrLn writer (input command)
            hFlush writer
            consume (Session (writer, reader, child) approve live) [] False
        _ -> ioError (userError "Worker protocol pipes were not created")

live :: ByteString -> IO ()
live line = Bytes.hPutStrLn stdout line >> hFlush stdout

consume :: Session problem -> [ByteString] -> Bool -> IO (Either (Failure problem) ByteString)
consume session collected granted = do
    let (_, reader, child) = pipes session
    ended <- hIsEOF reader
    if ended
        then complete child collected granted
        else Bytes.hGetLine reader >>= receive session (collected, granted)

receive :: Session problem -> ([ByteString], Bool) -> ByteString -> IO (Either (Failure problem) ByteString)
receive session (collected, granted) line = do
    emit session line
    let observed = line : collected
    case stage line of
        Left problem -> reject session (Protocol problem)
        Right "consumed" | granted -> reject session (Protocol "Duplicate consumption request")
        Right "consumed" -> authorize session observed
        Right _ -> consume session observed granted

authorize :: Session problem -> [ByteString] -> IO (Either (Failure problem) ByteString)
authorize session collected = do
    permitted <- review session (Bytes.unlines (reverse collected))
    case permitted of
        Left problem -> reject session (Rejected problem)
        Right permission -> do
            let (writer, _, _) = pipes session
            Bytes.hPutStrLn writer permission
            hFlush writer
            consume session collected True

reject :: Session problem -> Failure problem -> IO (Either (Failure problem) value)
reject session problem = do
    let (_, _, child) = pipes session
    terminateProcess child
    _ <- waitForProcess child
    pure (Left problem)

complete :: ProcessHandle -> [ByteString] -> Bool -> IO (Either (Failure problem) ByteString)
complete child collected granted = do
    status <- waitForProcess child
    pure $ case status of
        ExitSuccess | granted -> Right (Bytes.unlines (reverse collected))
        ExitSuccess -> Left (Protocol "Worker exited without approved consumption")
        _ -> Left (Exit status)

stage :: ByteString -> Either String String
stage encoded = eitherDecodeStrict encoded >>= parseEither (.: "stage")
