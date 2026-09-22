module Main (main) where

import Comparison qualified
import CycleReplay qualified
import Evaluation qualified
import InferenceBatch qualified
import InferenceInput qualified
import Inspection qualified
import Invar.Infer qualified as I
import Invar.Infer.Invocation qualified as Call
import Invar.Spec.Invocation qualified as V
import Invar.Worker qualified as W
import Options qualified as O
import Performance qualified
import PolicyInput qualified
import Quality qualified
import Scoring qualified
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Environment (getArgs)
import System.Exit (die)
import System.IO (BufferMode (LineBuffering), hSetBuffering, stdout)
import Training qualified
import Use qualified

main :: IO ()
main = do
    hSetBuffering stdout LineBuffering
    supplied <- getArgs
    case supplied of
        ["--help"] -> putStrLn usage
        ["infer", "--help"] -> putStrLn usage
        "evaluate" : rest -> Evaluation.run rest
        "quality" : rest -> Quality.run rest
        "performance" : rest -> Performance.run rest
        "inspect" : rest -> Inspection.run rest
        "compare" : rest -> Comparison.run rest
        "score" : rest -> Scoring.run rest
        "use" : rest -> Use.run rest
        "train" : rest -> Training.run rest
        "policy" : rest -> PolicyInput.run rest
        "replay" : rest -> CycleReplay.run rest
        "infer" : "batch" : rest -> InferenceBatch.run rest
        "infer" : rest -> do
            (worker, planned, bound) <- parse rest
            call <- either (die . show) pure (Call.prepare bound planned)
            W.run worker call >>= either (die . show) (const (pure ()))
        _ -> die usage

usage :: String
usage = usageInfo "Usage: invar use inspect|admit OPTIONS | invar infer OPTIONS | invar infer batch OPTIONS < calls.json | invar score OPTIONS | invar policy OPTIONS | invar train OPTIONS < tasks.json | invar evaluate OPTIONS < tasks.json | invar quality OPTIONS | invar performance OPTIONS | invar inspect KIND OPTIONS | invar compare KIND OPTIONS | invar replay KIND OPTIONS\nUse the command's --help for its options. Inference selects --checkpoint, or an explicit --adapter with all four digest options. --checkpoint reads the published policy description. --worker-config is optional." options

options :: [OptDescr (String, String)]
options = O.descriptions [("python", "Python executable"), ("worker", "Inference worker script"), ("worker-config", "Optional worker launch configuration"), ("cache", "Pinned model cache"), ("adapter", "Explicit adapter file or native handoff directory"), ("checkpoint", "Checkpoint containing policy.json and adapter.safetensors")] ++ InferenceInput.options

parse :: [String] -> IO (W.Worker, I.Plan, V.Binding)
parse supplied = do
    fields <- either die pure (O.parse options supplied)
    (adapter, planned) <- PolicyInput.plan fields
    worker <- either die pure (parseWorker fields adapter)
    bound <- either die pure (InferenceInput.binding fields)
    pure (worker, planned, bound)

parseWorker :: O.Fields -> FilePath -> Either String W.Worker
parseWorker fields adapter = do
    python <- O.required fields "python"
    worker <- O.required fields "worker"
    cache <- O.required fields "cache"
    pure W.Worker {W.executable = python, W.script = worker, W.cache = cache, W.adapter = adapter, W.environment = [], W.configuration = O.optional fields "worker-config"}
