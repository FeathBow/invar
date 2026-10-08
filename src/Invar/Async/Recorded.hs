module Invar.Async.Recorded (transcripts, generations) where

import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.Char (isDigit)
import Data.List (sort, stripPrefix)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (catMaybes)
import Invar.Artifact qualified as Artifact
import Invar.Async.Entry qualified as Entry
import Invar.Policy qualified as Policy
import Numeric.Natural (Natural)
import System.Directory (listDirectory)
import System.FilePath ((</>))
import System.IO.Error (isDoesNotExistError, tryIOError)

transcripts :: FilePath -> [Entry.Entry] -> IO (Map Natural ByteString)
transcripts root later = Map.fromList . catMaybes <$> traverse found [number | Entry.Reserved number _ _ _ <- later]
  where
    found number = do
        read' <- tryIOError (Bytes.readFile (root </> "transcripts" </> (show number ++ ".jsonl")))
        case read' of
            Right encoded -> pure (Just (number, encoded))
            Left problem
                | isDoesNotExistError problem -> pure Nothing
                | otherwise -> ioError problem

generations :: FilePath -> IO [Entry.Generation]
generations root = do
    names <- listDirectory root
    traverse generation (sort [read digits | name <- names, Just digits <- [stripPrefix "generation" name], not (null digits), all isDigit digits])
  where
    generation version = do
        let published = root </> ("generation" ++ show version)
        Entry.Generation version <$> Policy.identity (published </> "adapter.safetensors") <*> Artifact.identity "Learner checkpoint" (published </> "learner.pt") <*> Policy.readDescription (published </> "policy.json")
