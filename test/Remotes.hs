{-# LANGUAGE OverloadedStrings #-}

module Remotes (remotes) where

import Control.Concurrent (threadDelay)
import Control.Exception (IOException, try)
import Control.Monad (forM_, void)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value (..), decodeStrict, encode, object, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as Char
import Data.ByteString.Lazy qualified as Lazy
import Hedgehog
import Invar.Artifact qualified as Artifact
import Invar.Remote qualified as R
import Invar.Remote.Control qualified as C
import System.Directory (doesFileExist, getTemporaryDirectory)
import System.Exit (ExitCode (..))
import System.FilePath ((</>))
import System.IO (Handle, hClose, hPutStrLn)
import System.Posix.Signals (sigHUP, signalProcess)
import System.Posix.Temp (mkdtemp)
import System.Process (CreateProcess (..), ProcessHandle, StdStream (..), createProcess, proc, readProcessWithExitCode, waitForProcess)

remotes :: Group
remotes =
    Group
        "Remote wrapper"
        [ ("the worker starts only after its instance is published, and status reads absent, started, then ended", once barrier)
        , ("a program that cannot start is a launch failure", once missing)
        , ("stdout is forwarded unchanged, with a final line without a newline, and the outcome counts and digests it", once forwarded)
        , ("after the reader goes away the wrapper keeps draining, survives, and counts every byte the worker wrote", once dropped)
        , ("termination ends the worker's whole process group through the wrapper, which survives to record it", once grouped)
        , ("a worker that ignores termination is killed after the grace", once escalated)
        , ("a hangup leaves the wrapper running and the worker finishes normally", once hangup)
        , ("a request without a live wrapper is unconfirmed and signals nothing", once unconfirmed)
        , ("control files are refused when they disagree, carry another format or field, or appear out of order", once refused)
        ]
  where
    once = withTests 1 . property

workspace :: PropertyT IO FilePath
workspace = evalIO (getTemporaryDirectory >>= \temporary -> mkdtemp (temporary </> "invar-remote."))

-- | Starts `invar remote run` with the worker's stdin and stdout as pipes; the environment names the instance as D.
launch :: FilePath -> String -> [String] -> PropertyT IO (Handle, Handle, ProcessHandle)
launch root nonce command = do
    created <- evalIO (createProcess (proc "invar" (["remote", "run", "--instance", root, "--nonce", nonce, "--environment", "D=" ++ root, "--"] ++ command)) {std_in = CreatePipe, std_out = CreatePipe})
    case created of
        (Just input, Just output, _, handle) -> pure (input, output, handle)
        _ -> failure

observed :: FilePath -> PropertyT IO C.Observation
observed root = evalIO (R.status root) >>= evalEither

started :: FilePath -> PropertyT IO C.Instance
started root = do
    waitUntil (doesFileExist (root </> "instance.json"))
    current <- observed root
    case C.state current of
        C.Started chosen -> pure chosen
        other -> annotateShow other >> failure

waitUntil :: IO Bool -> PropertyT IO ()
waitUntil holds = go (250 :: Int)
  where
    go remaining = do
        done <- evalIO holds
        if done then pure () else if remaining <= 0 then failure else evalIO (threadDelay 20000) >> go (remaining - 1)

shell :: String -> [String]
shell script = ["/bin/sh", "-c", script]

ended :: FilePath -> PropertyT IO C.Outcome
ended root = do
    current <- observed root
    C.alive current === False
    case C.state current of
        C.Ended outcome -> pure outcome
        other -> annotateShow other >> failure

barrier :: PropertyT IO ()
barrier = do
    root <- (</> "instance") <$> workspace
    before <- observed root
    C.state before === C.Absent
    (input, output, handle) <- launch root "barrier" (shell "test -f \"$D/instance.json\" && printf published; read line; printf ' %s' \"$line\"")
    chosen <- started root
    during <- observed root
    C.alive during === True
    evalIO (hPutStrLn input "released" >> hClose input)
    evalIO (Bytes.hGetContents output) >>= (=== "published released")
    evalIO (waitForProcess handle) >>= (=== ExitSuccess)
    outcome <- ended root
    C.ended outcome === chosen
    C.ending outcome === C.Exited 0

missing :: PropertyT IO ()
missing = do
    root <- (</> "instance") <$> workspace
    (input, output, handle) <- launch root "missing" ["/nonexistent/program"]
    evalIO (hClose input >> void (Bytes.hGetContents output))
    evalIO (waitForProcess handle) >>= (=== ExitFailure 125)
    current <- observed root
    case C.state current of
        C.LaunchFailed (Just _) _ -> success
        other -> annotateShow other >> failure

forwarded :: PropertyT IO ()
forwarded = do
    root <- (</> "instance") <$> workspace
    (input, output, handle) <- launch root "forward" (shell "printf 'first\\nsecond\\nthird'; exit 3")
    evalIO (hClose input)
    received <- evalIO (Bytes.hGetContents output)
    received === "first\nsecond\nthird"
    evalIO (waitForProcess handle) >>= (=== ExitFailure 3)
    outcome <- ended root
    C.ending outcome === C.Exited 3
    (C.bytes outcome, C.digest outcome) === (fromIntegral (Bytes.length received), Artifact.hex (SHA256.hash received))

dropped :: PropertyT IO ()
dropped = do
    root <- (</> "instance") <$> workspace
    let size = 1048576 :: Int
    (input, output, handle) <- launch root "dropped" (shell ("head -c " ++ show size ++ " /dev/zero"))
    evalIO (hClose input)
    first <- evalIO (Bytes.hGetSome output 16)
    assert (not (Bytes.null first))
    evalIO (hClose output)
    evalIO (waitForProcess handle) >>= (=== ExitSuccess)
    outcome <- ended root
    (C.bytes outcome, C.digest outcome) === (fromIntegral size, Artifact.hex (SHA256.hash (Bytes.replicate size 0)))

-- | Reads the instance to terminate and runs `invar remote terminate` for it.
terminated :: FilePath -> C.Instance -> Int -> PropertyT IO Value
terminated root chosen grace = do
    let arguments = ["remote", "terminate", "--instance", root, "--nonce", "group", "--wrapper", show (C.wrapper (C.launching chosen)), "--worker", show (C.worker chosen), "--grace", show grace, "--wait", "5"]
    (code, printed, _) <- evalIO (readProcessWithExitCode "invar" arguments "")
    code === ExitSuccess
    maybe failure pure (decodeStrict (Char.pack printed))

gone :: Int -> IO Bool
gone pid = either (\(_ :: IOException) -> True) (const False) <$> try (signalProcess 0 (fromIntegral pid))

grouped :: PropertyT IO ()
grouped = do
    root <- (</> "instance") <$> workspace
    (input, _, handle) <- launch root "group" (shell "sleep 300 & echo $! > \"$D.grandchild\"; wait")
    chosen <- started root
    waitUntil (doesFileExist (root ++ ".grandchild"))
    grandchild <- evalIO (read . Char.unpack . Char.strip <$> Bytes.readFile (root ++ ".grandchild"))
    answer <- terminated root chosen 5
    answer === object ["requested" .= True, "confirmed" .= True]
    evalIO (hClose input)
    void (evalIO (waitForProcess handle))
    waitUntil (gone grandchild)
    outcome <- ended root
    C.ending outcome === C.Signalled 15
    current <- observed root
    fmap C.signal (C.terminated current) === Just C.Terminate
    C.killed current === Nothing

escalated :: PropertyT IO ()
escalated = do
    root <- (</> "instance") <$> workspace
    (input, _, handle) <- launch root "group" (shell "trap '' TERM; while :; do sleep 1; done")
    chosen <- started root
    answer <- terminated root chosen 1
    answer === object ["requested" .= True, "confirmed" .= True]
    evalIO (hClose input)
    void (evalIO (waitForProcess handle))
    outcome <- ended root
    C.ending outcome === C.Signalled 9
    current <- observed root
    fmap C.signal (C.killed current) === Just C.Kill

hangup :: PropertyT IO ()
hangup = do
    root <- (</> "instance") <$> workspace
    (input, output, handle) <- launch root "hangup" (shell "read line; printf '%s' \"$line\"")
    chosen <- started root
    evalIO (signalProcess sigHUP (fromIntegral (C.wrapper (C.launching chosen))))
    evalIO (threadDelay 200000)
    during <- observed root
    C.alive during === True
    evalIO (hPutStrLn input "finished" >> hClose input)
    evalIO (Bytes.hGetContents output) >>= (=== "finished")
    evalIO (waitForProcess handle) >>= (=== ExitSuccess)
    outcome <- ended root
    C.ending outcome === C.Exited 0

unconfirmed :: PropertyT IO ()
unconfirmed = do
    root <- (</> "instance") <$> workspace
    (input, output, handle) <- launch root "group" (shell "exit 0")
    chosen <- started root
    evalIO (hClose input >> void (Bytes.hGetContents output) >> void (waitForProcess handle))
    answer <- terminated root chosen 1
    answer === object ["requested" .= False, "confirmed" .= False]
    evalIO (doesFileExist (root </> "terminate.json")) >>= (=== False)

refused :: PropertyT IO ()
refused = do
    let launched = C.Launching "host" "nonce" 10
        other = C.Launching "host" "other" 10
        chosen = C.Instance launched 11
        outcome = C.Outcome chosen (C.Exited 0) 1 0 (Artifact.hex (SHA256.hash ""))
        sent signal = C.Sent chosen signal 1
        files = C.Files Nothing Nothing Nothing Nothing Nothing Nothing
        complete = files {C.launchingBytes = Just (C.launchingFile launched), C.instanceBytes = Just (C.instanceFile chosen), C.outcomeBytes = Just (C.outcomeFile outcome)}
        changed key value encoded = case decodeStrict encoded of
            Just (Object fields) -> Lazy.toStrict (encode (Object (Fields.insert key value fields)))
            _ -> encoded
        dropped' key encoded = case decodeStrict encoded of
            Just (Object fields) -> Lazy.toStrict (encode (Object (Fields.delete key fields)))
            _ -> encoded
    fmap C.state (C.observe False complete) === Right (C.Ended outcome)
    fmap C.state (C.observe True files {C.launchingBytes = Just (C.launchingFile launched)}) === Right (C.Launched launched)
    forM_
        [ complete {C.instanceBytes = Nothing}
        , complete {C.launchingBytes = Nothing}
        , complete {C.instanceBytes = Just (C.instanceFile (C.Instance other 11))}
        , complete {C.failureBytes = Just (C.failureFile (C.Failure launched "failed"))}
        , complete {C.killedBytes = Just (C.signalledFile (sent C.Kill))}
        , complete {C.terminatedBytes = Just (C.signalledFile (sent C.Kill))}
        , complete {C.terminatedBytes = Just (C.signalledFile (C.Sent (C.Instance launched 12) C.Terminate 1))}
        , complete {C.instanceBytes = Just (changed "group" (Number 12) (C.instanceFile chosen))}
        , complete {C.instanceBytes = Just (changed "format" (String "invar-remote-launching-v1") (C.instanceFile chosen))}
        , complete {C.instanceBytes = Just (changed "extra" Null (C.instanceFile chosen))}
        , complete {C.outcomeBytes = Just (dropped' "sha256" (C.outcomeFile outcome))}
        , complete {C.outcomeBytes = Just (changed "ending" (object ["exited" .= (0 :: Int), "signalled" .= (9 :: Int)]) (C.outcomeFile outcome))}
        ]
        $ \supplied -> case C.observe False supplied of
            Left _ -> success
            Right accepted -> annotateShow accepted >> failure
    C.request (C.requestFile (C.Request "nonce" 10 11 2)) === Right (C.Request "nonce" 10 11 2)
