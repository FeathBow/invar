module UseInput (readContract, readDeclared) where

import Control.Monad (unless)
import Data.Aeson (Value, eitherDecodeStrict)
import Data.ByteString qualified as Bytes
import Envelope (Problem (..), refuse)
import Invar.Use qualified as U
import System.Directory (doesFileExist)

readContract :: String -> String -> FilePath -> IO (U.UseContract, Bytes.ByteString)
readContract format name path = do
    bytes <- present format name path
    either (\found -> refuse format [Problem "artifact-invalid" ("artifact:" ++ path) found]) (\contract -> pure (contract, bytes)) (U.decodeContract bytes)

readDeclared :: String -> String -> FilePath -> IO (Value, Bytes.ByteString)
readDeclared format name path = do
    bytes <- present format name path
    either (\found -> refuse format [Problem "artifact-invalid" ("artifact:" ++ path) found]) (\parsed -> pure (parsed, bytes)) (eitherDecodeStrict bytes)

present :: String -> String -> FilePath -> IO Bytes.ByteString
present format name path = do
    found <- doesFileExist path
    unless found (refuse format [Problem "artifact-missing" ("argv:--" ++ name) (path ++ " does not exist")])
    Bytes.readFile path
