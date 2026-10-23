{-# LANGUAGE OverloadedStrings #-}

module Invar.Remote (Launch (..), run, exec, status, terminate) where

import Control.Concurrent (MVar, forkIO, newEmptyMVar, putMVar, takeMVar, threadDelay)
import Control.Exception (IOException, try)
import Control.Monad (unless, void)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value, object, (.=))
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as Char
import Data.Maybe (isJust)
import Data.Text (Text)
import Data.Text qualified as Text
import GHC.Clock (getMonotonicTime)
import Invar.Artifact qualified as Artifact
import Invar.Remote.Control qualified as C
import Invar.Store qualified as Store
import Numeric.Natural (Natural)
import System.Directory (createDirectory, doesDirectoryExist, doesFileExist)
import System.Exit (ExitCode (..), exitWith)
import System.FilePath ((</>))
import System.IO (Handle, SeekMode (AbsoluteSeek), hClose, hFlush, hSetBinaryMode, stdout)
import System.IO.Error (catchIOError, isEOFError)
import System.Posix.IO qualified as Posix
import System.Posix.IO.ByteString qualified as Raw
import System.Posix.Process (executeFile, getProcessID)
import System.Posix.Signals (Handler (Default, Ignore), Signal, installHandler, killProcess, sigHUP, sigPIPE, signalProcessGroup, softwareTermination)
import System.Posix.Time (epochTime)
import System.Posix.Types (Fd)
import System.Posix.Unistd (getSystemID, nodeName)
import System.Process (CreateProcess (..), ProcessHandle, StdStream (..), createProcess, getPid, getProcessExitCode, proc)

-- | A remote launch: the instance directory and nonce the core chose, and the worker with exactly the declared
-- working directory and environment.
data Launch = Launch {instanceDirectory :: FilePath, launchNonce :: Text, directory :: Maybe FilePath, environment :: [(String, String)], program :: FilePath, arguments :: [String]}

-- | Runs the worker behind the start barrier as a child of this process, forwards its stdout unchanged, and is the
-- only owner that signals and reaps it. `self` is this executable, started again as `invar remote exec`.
run :: FilePath -> Launch -> IO ExitCode
run self launch = do
    let root = instanceDirectory launch
    createDirectory root
    held <- Posix.openFd (root </> "lock") Posix.ReadWrite Posix.defaultFileFlags {Posix.creat = Just 0o644, Posix.exclusive = True, Posix.cloexec = True}
    Posix.setLock held (Posix.WriteLock, AbsoluteSeek, 0, 0)
    name <- Text.pack . nodeName <$> getSystemID
    own <- fromIntegral <$> getProcessID
    let launched = C.Launching name (launchNonce launch) own
    publish root "launching" (C.launchingFile launched)
    void (installHandler sigHUP Ignore Nothing)
    (barrier, release) <- Posix.createPipe
    (report, reported) <- Posix.createPipe
    Posix.setFdOption release Posix.CloseOnExec True
    Posix.setFdOption report Posix.CloseOnExec True
    let child = (proc self (["remote", "exec", "--barrier", show barrier, "--status", show reported, "--", program launch] ++ arguments launch)) {std_in = Inherit, std_out = CreatePipe, std_err = Inherit, cwd = directory launch, env = Just (environment launch), create_group = True}
    spawned <- try (createProcess child)
    Posix.closeFd barrier
    Posix.closeFd reported
    case spawned of
        Left (problem :: IOException) -> do
            publish root "failure" (C.failureFile (C.Failure launched (Text.pack (show problem))))
            pure (ExitFailure 125)
        Right (_, Just output, _, handle) -> do
            pid <- getPid handle >>= maybe (ioError (userError "The wrapper's child has no process id")) pure
            let started = C.Instance launched (fromIntegral pid)
            publish root "instance" (C.instanceFile started)
            void (Raw.fdWrite release "1")
            Posix.closeFd release
            forwarded <- forward output
            failure <- Posix.fdToHandle report >>= \source -> Bytes.hGetContents source
            code <- supervise root started handle
            (count, sha) <- takeMVar forwarded
            at <- epochTime
            if Bytes.null failure
                then do
                    publish root "outcome" (C.outcomeFile (C.Outcome started (ending code) (fromIntegral (fromEnum at)) count sha))
                    pure code
                else do
                    publish root "failure" (C.failureFile (C.Failure launched (Text.pack (Char.unpack failure))))
                    pure (ExitFailure 125)
        Right _ -> ioError (userError "The worker's stdout pipe was not created")
  where
    ending ExitSuccess = C.Exited 0
    ending (ExitFailure code)
        | code < 0 = C.Signalled (negate code)
        | otherwise = C.Exited code

-- | Forwards every byte of the worker's stdout to this process's stdout, and counts and digests all of it. When a
-- write fails because the connection dropped, it stops writing and keeps reading to the end.
forward :: Handle -> IO (MVar (Natural, String))
forward source = do
    done <- newEmptyMVar
    hSetBinaryMode source True
    hSetBinaryMode stdout True
    let go context count open = do
            chunk <- Bytes.hGetSome source 65536
            if Bytes.null chunk
                then hClose source >> putMVar done (count, Artifact.hex (SHA256.finalize context))
                else do
                    writing <- if open then (Bytes.hPut stdout chunk >> hFlush stdout >> pure True) `catchIOError` const (pure False) else pure False
                    go (SHA256.update context chunk) (count + fromIntegral (Bytes.length chunk)) writing
    void (forkIO (go SHA256.init 0 True))
    pure done

data Phase = Waiting | Ignored | Terminating Double | Killed

-- | The one loop that owns the child. It reaps it, and while it still holds the unreaped leader it answers a
-- matching termination request by signalling the worker's process group, then kills the group after the grace.
supervise :: FilePath -> C.Instance -> ProcessHandle -> IO ExitCode
supervise root started handle = go Waiting
  where
    go phase = do
        exited <- getProcessExitCode handle
        case exited of
            Just code -> pure code
            Nothing -> advance phase >>= \next -> threadDelay 20000 >> go next
    advance Waiting = do
        pending <- doesFileExist (root </> "terminate.json")
        if not pending
            then pure Waiting
            else do
                decoded <- C.request <$> Bytes.readFile (root </> "terminate.json")
                case decoded of
                    Right chosen | matches chosen -> do
                        sent <- send softwareTermination C.Terminate "terminated"
                        now <- getMonotonicTime
                        pure (if sent then Terminating (now + fromIntegral (C.grace chosen)) else Ignored)
                    _ -> pure Ignored
    advance (Terminating deadline) = do
        now <- getMonotonicTime
        if now < deadline then pure (Terminating deadline) else Killed <$ send killProcess C.Kill "killed"
    advance other = pure other
    matches chosen = C.requested chosen == C.nonce (C.launching started) && C.requestedWrapper chosen == C.wrapper (C.launching started) && C.requestedWorker chosen == C.worker started
    send :: Signal -> C.Signal -> String -> IO Bool
    send number named record = do
        delivered <- try (signalProcessGroup number (fromIntegral (C.worker started)))
        case delivered of
            Left (_ :: IOException) -> pure False
            Right () -> do
                at <- epochTime
                publish root record (C.signalledFile (C.Sent started named (fromIntegral (fromEnum at))))
                pure True

-- | The child started by `run`: it leads the process group it was created in, waits for the barrier, and replaces
-- itself with the worker. A failed replacement is written to the status descriptor, which is closed on success.
exec :: Fd -> Fd -> FilePath -> [String] -> IO ()
exec barrier report target values = do
    released <- (not . Bytes.null <$> Raw.fdRead barrier 1) `catchIOError` \problem -> if isEOFError problem then pure False else ioError problem
    Posix.closeFd barrier
    unless released (exitWith (ExitFailure 125))
    Posix.setFdOption report Posix.CloseOnExec True
    void (installHandler sigHUP Default Nothing)
    void (installHandler sigPIPE Default Nothing)
    executeFile target False values Nothing `catchIOError` \problem -> do
        void (Raw.fdWrite report (Char.pack (show problem)))
        exitWith (ExitFailure 127)

-- | What the wrapper wrote so far, read in the reverse order of writing so that every file read names files that
-- already existed, and whether a live wrapper still holds the instance.
status :: FilePath -> IO (Either String C.Observation)
status root = do
    present <- doesDirectoryExist root
    if not present
        then pure (C.observe False (C.Files Nothing Nothing Nothing Nothing Nothing Nothing))
        else do
            holding <- alive root
            outcome <- optional "outcome.json"
            failure <- optional "failure.json"
            killed <- optional "killed.json"
            terminated <- optional "terminated.json"
            published <- optional "instance.json"
            launched <- optional "launching.json"
            pure (C.observe holding (C.Files launched published failure outcome terminated killed))
  where
    optional name = do
        exists <- doesFileExist (root </> name)
        if exists then Just <$> Bytes.readFile (root </> name) else pure Nothing

alive :: FilePath -> IO Bool
alive root = do
    exists <- doesFileExist (root </> "lock")
    if not exists
        then pure False
        else do
            file <- Posix.openFd (root </> "lock") Posix.ReadWrite Posix.defaultFileFlags {Posix.cloexec = True}
            conflict <- Posix.getLock file (Posix.WriteLock, AbsoluteSeek, 0, 0)
            Posix.closeFd file
            pure (isJust conflict)

-- | Asks a live wrapper to terminate its worker and waits for its confirmation. Without a live wrapper nothing is
-- written and nothing is signalled.
terminate :: FilePath -> C.Request -> Int -> IO Value
terminate root chosen seconds = do
    holding <- alive root
    if not holding
        then pure (object ["requested" .= False, "confirmed" .= False])
        else do
            publish root "terminate" (C.requestFile chosen)
            confirmed <- waitFor (seconds * 50) (doesFileExist (root </> "terminated.json"))
            pure (object ["requested" .= True, "confirmed" .= confirmed])
  where
    waitFor remaining check = do
        done <- check
        if done || remaining <= 0 then pure done else threadDelay 20000 >> waitFor (remaining - 1 :: Int) check

-- | Writes a control file under a staging name, synchronizes it, renames it into place without replacing anything,
-- and synchronizes the directory.
publish :: FilePath -> String -> ByteString -> IO ()
publish root name encoded = do
    let staged = name ++ ".staging"
    file <- Posix.openFd (root </> staged) Posix.WriteOnly Posix.defaultFileFlags {Posix.creat = Just 0o644, Posix.exclusive = True, Posix.cloexec = True}
    written <- Posix.fdToHandle file
    Bytes.hPut written encoded
    hClose written
    void (Store.publish (Store.Location root (Char.pack staged) (Char.pack (name ++ ".json"))))
