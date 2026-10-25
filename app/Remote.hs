{-# LANGUAGE OverloadedStrings #-}

module Remote (run) where

import Control.Monad (unless)
import Data.Aeson (encode, object, (.=))
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Data.List (nub)
import Data.Text qualified as Text
import Invar.Remote qualified as R
import Invar.Remote.Control qualified as C
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Environment (getExecutablePath)
import System.Exit (die, exitWith)

run :: [String] -> IO ()
run ["--help"] = putStrLn usage
run [_, "--help"] = putStrLn usage
run ("run" : supplied) = do
    let (named, command) = break (== "--") supplied
        (declared, rest) = environment named
    (target, values) <- case drop 1 command of
        chosen : remaining -> pure (chosen, remaining)
        [] -> die "remote run needs -- PROGRAM [ARGUMENT]..."
    fields <- either die pure (O.parse runOptions rest)
    pairs <- either die pure (traverse assignment declared)
    unless (length (nub (map fst pairs)) == length pairs) (die "Duplicate environment names are not allowed")
    root <- either die pure (O.required fields "instance")
    chosen <- either die pure (O.required fields "nonce")
    self <- getExecutablePath
    R.run self (R.Launch root (Text.pack chosen) (O.optional fields "directory") pairs target values) >>= exitWith
run ("exec" : supplied) = do
    let (named, command) = break (== "--") supplied
    fields <- either die pure (O.parse execOptions named)
    barrier <- either die pure (O.numeric fields "barrier")
    report <- either die pure (O.numeric fields "status")
    case drop 1 command of
        target : values -> R.exec (fromInteger barrier) (fromInteger report) target values
        [] -> die "remote exec needs -- PROGRAM [ARGUMENT]..."
run ("status" : supplied) = do
    fields <- either die pure (O.parse statusOptions supplied)
    root <- either die pure (O.required fields "instance")
    R.status root >>= either die (Lazy.putStrLn . encode . C.describe)
run ("terminate" : supplied) = do
    fields <- either die pure (O.parse terminateOptions supplied)
    root <- either die pure (O.required fields "instance")
    chosen <- either die pure (O.required fields "nonce")
    request <- either die pure (C.Request (Text.pack chosen) <$> O.numeric fields "wrapper" <*> O.numeric fields "worker" <*> O.numeric fields "grace")
    seconds <- either die pure (O.numeric fields "wait")
    (written, answered, current) <- R.terminate root request seconds
    Lazy.putStrLn (encode (object ["requested" .= written, "answer" .= C.answerName answered, "status" .= C.describe current]))
run _ = die usage

environment :: [String] -> ([String], [String])
environment ("--environment" : value : rest) = let (declared, remaining) = environment rest in (value : declared, remaining)
environment (other : rest) = fmap (other :) (environment rest)
environment [] = ([], [])

assignment :: String -> Either String (String, String)
assignment value = case break (== '=') value of
    (name@(_ : _), '=' : setting) -> Right (name, setting)
    _ -> Left ("An environment entry is NAME=VALUE: " ++ value)

runOptions, execOptions, statusOptions, terminateOptions :: [OptDescr (String, String)]
runOptions = O.descriptions [("instance", "Instance directory to create; it must not exist"), ("nonce", "Nonce the core chose for this launch"), ("directory", "Working directory of the worker")]
execOptions = O.descriptions [("barrier", "Descriptor to read the release from"), ("status", "Descriptor to report a failed start on")]
statusOptions = O.descriptions [("instance", "Instance directory")]
terminateOptions = O.descriptions [("instance", "Instance directory"), ("nonce", "Nonce of the instance"), ("wrapper", "Wrapper process id of the instance"), ("worker", "Worker process id of the instance"), ("grace", "Seconds between termination and kill"), ("wait", "Seconds to wait for the wrapper's confirmation")]

usage :: String
usage =
    usageInfo
        "Usage: invar remote run --instance DIRECTORY --nonce NONCE [--directory PATH] [--environment NAME=VALUE]... -- PROGRAM [ARGUMENT]...\n       invar remote status --instance DIRECTORY\n       invar remote terminate --instance DIRECTORY --nonce NONCE --wrapper PID --worker PID --grace SECONDS --wait SECONDS\nrun starts the worker behind a start barrier with exactly the declared environment, forwards its stdout unchanged, and writes its control files in the instance directory; status and terminate print one JSON object. invar remote exec is the internal child of run."
        (runOptions ++ terminateOptions)
