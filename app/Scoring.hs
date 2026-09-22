module Scoring (run) where

import Data.Aeson (eitherDecodeStrict, encode)
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as TextBytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import InferenceInput qualified
import Invar.Infer qualified as Infer
import Invar.Infer.Observation qualified as Inference
import Invar.Infer.Result qualified as Result
import Invar.Score qualified as Score
import Invar.Score.Worker qualified as Execution
import Invar.Worker qualified as Worker
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)

run :: [String] -> IO ()
run ["--help"] = putStrLn usage
run ["plan", "--help"] = putStrLn usage
run ["inspect", "--help"] = putStrLn usage
run ("plan" : supplied) = do
    fields <- either die pure (O.parse common supplied)
    prepare fields >>= TextBytes.putStrLn . Score.input
run ("inspect" : supplied) = do
    fields <- either die pure (O.parse (common ++ inspection) supplied)
    call <- prepare fields
    path <- either die pure (O.required fields "log")
    status <- either die pure (O.numeric fields "exit-code")
    bytes <- Bytes.readFile path
    result <- either (die . show) pure (Score.admit call status bytes)
    Lazy.putStrLn (encode (Score.describe result))
run supplied = do
    fields <- either die pure (O.parse (common ++ execution) supplied)
    call <- prepare fields
    python <- either die pure (O.required fields "python")
    script <- either die pure (O.required fields "worker")
    cache <- either die pure (O.required fields "cache")
    adapter <- either die pure (O.required fields "adapter")
    let worker = Worker.Worker python script cache adapter [] (O.optional fields "worker-config")
    Execution.run worker call >>= either (die . show) (const (pure ()))

prepare :: O.Fields -> IO Score.Call
prepare fields = do
    original <- either die pure (InferenceInput.requestWith "source-" fields)
    sourcePlan <- either (die . show) pure (Infer.prepare original)
    sourceBinding <- either die pure (InferenceInput.bindingWith "source-" fields)
    sourcePath <- either die pure (O.required fields "source-log")
    sourceStatus <- either die pure (O.numeric fields "source-exit-code")
    encoded <- Bytes.readFile sourcePath
    source <- either die pure (Inference.admit sourcePlan sourceBinding encoded)
    adapter <- either die pure (O.required fields "target-digest")
    tokenizer <- either die pure (O.required fields "target-tokenizer-digest")
    base <- either die pure (O.required fields "target-base-digest")
    assembly <- either die pure (O.required fields "target-assembly-digest")
    let requested = (Result.consumed (Inference.result source)) {Infer.artifact = adapter, Infer.tokenizer = tokenizer, Infer.base = base, Infer.assembly = assembly}
    selected <- either (die . show) pure (Infer.prepare requested)
    basePlan <- either (die . show) pure (Score.prepare sourceStatus source selected)
    planned <- case O.optional fields "probe-steps" of
        Nothing -> pure basePlan
        Just selection -> do
            steps <- either die pure (eitherDecodeStrict (TextBytes.pack selection))
            either (die . show) pure (Score.withProbe steps basePlan)
    bound <- either die pure (InferenceInput.binding fields)
    either (die . show) pure (Score.bind bound planned)

common :: [OptDescr (String, String)]
common =
    InferenceInput.optionsWith "source-"
        ++ O.descriptions
            [ ("source-log", "Complete source free-generation log")
            , ("source-exit-code", "Independently recorded source process exit status")
            , ("target-digest", "Actual target adapter identity")
            , ("target-tokenizer-digest", "Actual target tokenizer identity")
            , ("target-base-digest", "Actual target base identity")
            , ("target-assembly-digest", "Actual target assembly identity")
            , ("call", "Score call identity")
            , ("attempt", "Score attempt identity")
            , ("instance", "Score activation instance")
            , ("probe-steps", "Optional nonempty JSON array of increasing response steps for full-vocabulary capture")
            ]

execution :: [OptDescr (String, String)]
execution = O.descriptions [("python", "Python executable"), ("worker", "Bound score worker script"), ("cache", "Local pinned model cache"), ("adapter", "Actual target adapter file"), ("worker-config", "Optional native configuration file")]

inspection :: [OptDescr (String, String)]
inspection = O.descriptions [("log", "Complete bound score stdout"), ("exit-code", "Independently recorded score process exit status")]

usage :: String
usage = usageInfo "Usage: invar score OPTIONS | invar score plan OPTIONS | invar score inspect OPTIONS\nScore the source response through the target's own native caches after checked consumption.\nPlan emits a bound worker input; inspect validates complete execution and emits a finite path ratio.\nNo KL or use permission is established." (common ++ execution ++ inspection)
