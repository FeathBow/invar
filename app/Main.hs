module Main (main) where

import Evaluation qualified
import Invar.Infer qualified as I
import Invar.Infer.Invocation qualified as Call
import Invar.Spec.Invocation qualified as V
import Invar.Worker qualified as W
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Environment (getArgs)
import System.Exit (die)
import Training qualified

main :: IO ()
main = do
    supplied <- getArgs
    case supplied of
        ["--help"] -> putStrLn usage
        ["infer", "--help"] -> putStrLn usage
        "evaluate" : rest -> Evaluation.run rest
        "train" : rest -> Training.run rest
        "infer" : rest -> do
            (worker, request, bound) <- either die pure (parse rest)
            planned <- either (die . show) pure (I.prepare request)
            call <- either (die . show) pure (Call.prepare bound planned)
            W.run worker call >>= either (die . show) (const (pure ()))
        _ -> die usage

usage :: String
usage = usageInfo "Usage: invar infer OPTIONS | invar train OPTIONS < tasks.json | invar evaluate OPTIONS < tasks.json\nUse the command's --help for its options. All inference options are required. Execution does not imply numerical certification." options

options :: [OptDescr (String, String)]
options = O.descriptions descriptions
  where
    descriptions = [("python", "Python executable"), ("worker", "Inference worker script"), ("cache", "Pinned model cache"), ("adapter", "Adapter file"), ("digest", "Canonical adapter tensor SHA-256"), ("tokenizer-digest", "Tokenizer operation SHA-256"), ("base-digest", "Frozen model tensor SHA-256"), ("assembly-digest", "Model assembly SHA-256"), ("prompt", "Input prompt"), ("tokens", "Positive token limit"), ("temperature", "Positive sampling temperature"), ("seed", "Logical sample seed"), ("call", "Logical call identity"), ("attempt", "Dispatch attempt identity"), ("instance", "Executor load-instance identity")]

parse :: [String] -> Either String (W.Worker, I.Request, V.Binding)
parse supplied = do
    fields <- O.parse options supplied
    worker <- parseWorker fields
    request <- parseRequest fields
    bound <- V.Binding . V.CallId <$> O.numeric fields "call" <*> (V.AttemptId <$> O.numeric fields "attempt") <*> (V.Instance <$> O.numeric fields "instance")
    pure (worker, request, bound)

parseWorker :: O.Fields -> Either String W.Worker
parseWorker fields = do
    python <- O.required fields "python"
    worker <- O.required fields "worker"
    cache <- O.required fields "cache"
    adapter <- O.required fields "adapter"
    pure W.Worker {W.executable = python, W.script = worker, W.cache = cache, W.adapter = adapter, W.environment = []}

parseRequest :: O.Fields -> Either String I.Request
parseRequest fields = do
    artifact <- O.required fields "digest"
    tokenizer <- O.required fields "tokenizer-digest"
    base <- O.required fields "base-digest"
    assembly <- O.required fields "assembly-digest"
    prompt <- O.required fields "prompt"
    tokens <- O.numeric fields "tokens"
    temperature <- O.numeric fields "temperature"
    seed <- O.numeric fields "seed"
    pure I.Request {I.artifact = artifact, I.tokenizer = tokenizer, I.base = base, I.assembly = assembly, I.prompt = prompt, I.tokens = tokens, I.temperature = temperature, I.seed = seed}
