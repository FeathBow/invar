module UpdateReplayInput (run) where

import Data.Aeson (encode)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Invar.Replay.Update qualified as Update
import Invar.Replay.Update.Output qualified as Output
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Directory (doesDirectoryExist)
import System.Exit (die)

run :: String -> [String] -> IO ()
run kind ["--help"] = putStrLn (usageInfo ("Usage: invar inspect " ++ kind ++ " OPTIONS") (options kind))
run "update-output" supplied = do
    fields <- either die pure (O.parse (options "update-output") supplied)
    resident <- either die pure (mode fields)
    snapshot <- Bytes.getContents
    path <- either die pure (O.required fields "log")
    staged <- either die pure (O.required fields "output")
    status <- either die pure (O.numeric fields "exit-code")
    encoded <- Bytes.readFile path
    observed <- case resident of
        Update.Resident -> do
            expected <- either die pure (Update.decodeMany snapshot)
            Output.inspectResident (expected, staged) (status, encoded)
        Update.Shared -> do
            expected <- either die pure (Update.decodeMany snapshot)
            Output.inspectShared (expected, staged) (status, encoded)
        Update.Finite -> do
            expected <- either die pure (Update.decodeBytes snapshot)
            Output.describe <$> Output.inspect (expected, staged) (status, encoded)
    Lazy.putStrLn (encode observed)
run kind supplied = do
    fields <- either die pure (O.parse (options kind) supplied)
    resident <- either die pure (mode fields)
    initial <- either die pure (O.required fields "initial")
    count <- either die pure (O.numeric fields "updates")
    status <- either die pure (O.numeric fields "exit-code")
    path <- either die pure (O.required fields "log")
    encoded <- Bytes.readFile path
    let admit = case resident of
            Update.Finite -> Update.admit
            Update.Resident -> Update.admitResident
            Update.Shared -> Update.admitShared
    observed <- admit doesDirectoryExist (Update.Run initial count status) encoded
    Lazy.putStrLn (encode (Update.describe observed))

options :: String -> [OptDescr (String, String)]
options kind = O.descriptions ([("exit-code", "Independently observed process exit status"), ("log", "Complete process stdout"), ("mode", "finite (default), resident or shared numerical process")] ++ selected)
  where
    selected = if kind == "update-output" then [("output", "Staged checkpoint, or resident checkpoint directory; expected snapshot(s) on stdin")] else [("initial", "Initial input checkpoint directory"), ("updates", "Independently declared positive update count")]

mode :: O.Fields -> Either String Update.Mode
mode fields = case O.optional fields "mode" of
    Nothing -> Right Update.Finite
    Just "finite" -> Right Update.Finite
    Just "resident" -> Right Update.Resident
    Just "shared" -> Right Update.Shared
    _ -> Left "Update replay mode must be finite, resident or shared"
