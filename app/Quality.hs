module Quality (run) where

import Data.Aeson (encode)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import EvaluationInput qualified
import Invar.Evaluation qualified as Evaluation
import Invar.Quality qualified as Quality
import Invar.Workload qualified as Workload
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)

run :: [String] -> IO ()
run ["--help"] = putStrLn usage
run supplied = do
    fields <- either die pure (O.parse options supplied)
    path <- either die pure (O.required fields "tasks")
    expected <- Bytes.readFile path >>= either die pure . Workload.decode
    first <- readReport fields expected "initial"
    second <- readReport fields expected "trained"
    report <- either die pure (Quality.compare first second)
    Lazy.putStrLn (encode report)

readReport :: O.Fields -> Workload.Document -> String -> IO Evaluation.Report
readReport fields expected side = do
    path <- either die pure (O.required fields (side ++ "-log"))
    selected <- EvaluationInput.declared (side ++ "-") fields
    encoded <- Bytes.readFile path
    either die pure (Evaluation.admit expected selected encoded)

options :: [OptDescr (String, String)]
options = O.descriptions [("tasks", "Identical frozen input used by both evaluations")] ++ concatMap (EvaluationInput.options . (++ "-")) ["initial", "trained"]

usage :: String
usage = usageInfo "Usage: invar quality OPTIONS\nCompare complete initial and trained-policy evaluation reports." options
