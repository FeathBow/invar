module Performance (run) where

import Data.Aeson (encode)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Invar.Measurement qualified as Measurement
import Invar.Workload qualified as Workload
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Directory (canonicalizePath)
import System.Exit (die)

run :: [String] -> IO ()
run ["--help"] = putStrLn (usageInfo "Usage: invar performance OPTIONS" options)
run supplied = do
    fields <- either die pure (O.parse options supplied)
    taskPath <- either die pure (O.required fields "tasks")
    tasks <- Bytes.readFile taskPath >>= either die pure . Workload.decode
    manifest <- either die pure (O.required fields "manifest")
    policy <- either die pure (O.required fields "policy")
    let source = Measurement.Source {Measurement.readBytes = Bytes.readFile, Measurement.resolvePath = canonicalizePath}
    observed <- Measurement.admit source tasks (policy, manifest)
    Lazy.putStrLn (encode observed)

options :: [OptDescr (String, String)]
options = O.descriptions [("manifest", "Complete repeated-run declaration with independent exits and elapsed durations"), ("tasks", "Identical frozen workload"), ("policy", "Expected reference policy identity")]
