module InferenceBatch (run) where

import Control.Monad (when)
import Data.Aeson (eitherDecodeStrict)
import Data.Bifunctor (first)
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as TextBytes
import InferenceInput qualified
import Invar.Infer qualified as Infer
import Invar.Infer.Invocation qualified as Call
import Invar.Worker qualified as Worker
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)
import System.IO (hFlush, stdout)

run :: [String] -> IO ()
run ["--help"] = putStrLn usage
run supplied = do
    fields <- either die pure (O.parse options supplied)
    worker <- either die pure (selected fields)
    encoded <- Bytes.getContents
    arguments <- either die pure (eitherDecodeStrict encoded)
    when (null arguments) (die "A finite inference batch requires at least one call")
    calls <- either die pure (traverse prepare arguments)
    Worker.runBatchedSession worker emit calls >>= either (die . show) (const (pure ()))
  where
    emit line = TextBytes.hPutStrLn stdout line >> hFlush stdout

prepare :: [String] -> Either String Call.Call
prepare supplied = do
    fields <- O.parse InferenceInput.options supplied
    requested <- InferenceInput.request fields
    planned <- first show (Infer.prepare requested)
    bound <- InferenceInput.binding fields
    first show (Call.prepare bound planned)

selected :: O.Fields -> Either String Worker.Worker
selected fields = Worker.Worker <$> string "python" <*> string "worker" <*> string "cache" <*> string "adapter" <*> pure [] <*> pure (O.optional fields "worker-config")
  where
    string = O.required fields

options :: [OptDescr (String, String)]
options = O.descriptions [("python", "Python executable"), ("worker", "Finite batch worker script"), ("cache", "Pinned model cache"), ("adapter", "Explicit adapter location shared by the batch"), ("worker-config", "Optional worker configuration")]

usage :: String
usage = usageInfo "Usage: invar infer batch OPTIONS < calls.json\nInput is a nonempty JSON array of unprefixed inference request/binding argument arrays.\nEvery call uses the shared adapter with explicit materialization digests.\nThe original grouped stdout is the source for each member's numerical observation." options
