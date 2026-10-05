module InferenceBatch (run) where

import Data.ByteString qualified as Bytes
import InferenceInput qualified
import Invar.Transcript qualified as Transcript
import Invar.Worker qualified as Worker
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)

run :: [String] -> IO ()
run ["--help"] = putStrLn usage
run supplied = do
    fields <- either die pure (O.parse options supplied)
    worker <- either die pure (selected fields)
    calls <- Bytes.getContents >>= either die pure . InferenceInput.calls
    Worker.runBatchedSession worker Nothing Transcript.standard calls >>= either (die . show) (const (pure ()))

selected :: O.Fields -> Either String Worker.Worker
selected fields = Worker.Worker <$> string "python" <*> string "worker" <*> string "cache" <*> string "adapter" <*> pure [] <*> pure (O.optional fields "worker-config")
  where
    string = O.required fields

options :: [OptDescr (String, String)]
options = O.descriptions [("python", "Python executable"), ("worker", "Finite batch worker script"), ("cache", "Pinned model cache"), ("adapter", "Explicit adapter location shared by the batch"), ("worker-config", "Optional worker configuration")]

usage :: String
usage = usageInfo "Usage: invar infer batch OPTIONS < calls.json\nInput is a nonempty JSON array of unprefixed inference request/binding argument arrays.\nEvery call uses the shared adapter with explicit materialization digests.\nThe original grouped stdout is the source for each member's numerical observation." options
