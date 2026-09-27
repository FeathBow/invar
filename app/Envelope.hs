{-# LANGUAGE OverloadedStrings #-}

module Envelope (Problem (..), command, output, succeed, refuse, rational) where

import Control.Exception (SomeException, catch, displayException, fromException, throwIO)
import Control.Monad (unless, when)
import Data.Aeson (Value, encode, object, (.=))
import Data.Aeson.Key qualified as Key
import Data.Aeson.Types (Pair)
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Data.Ratio (denominator, numerator)
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Directory (doesDirectoryExist, doesPathExist)
import System.Exit (ExitCode (..), exitWith)
import System.FilePath (takeDirectory)

data Problem = Problem {code :: String, at :: String, message :: String}

command :: String -> String -> [OptDescr (String, String)] -> ([String] -> IO ()) -> [String] -> IO ()
command _ header options _ ["--help"] = putStrLn (usageInfo header options)
command format _ _ body supplied =
    body supplied `catch` \failure -> case fromException failure of
        Just exit -> throwIO (exit :: ExitCode)
        Nothing -> refuse format [Problem "internal-error" "argv" ("Unexpected failure: " ++ displayException (failure :: SomeException))]

output :: String -> String -> FilePath -> IO ()
output format written path = do
    exists <- doesPathExist path
    when exists (refuse format [Problem "invalid-value" "argv:--output" (path ++ " already exists; " ++ written ++ " is never written over")])
    parent <- doesDirectoryExist (takeDirectory path)
    unless parent (refuse format [Problem "invalid-value" "argv:--output" ("The directory that would hold " ++ path ++ " does not exist")])

succeed :: String -> String -> String -> String -> [Pair] -> IO ()
succeed format status means doesNotMean fields =
    Lazy.putStrLn (encode (object (["format" .= format, "status" .= status, "means" .= means, "does_not_mean" .= doesNotMean, "evidence" .= False, "problems" .= ([] :: [Value])] ++ fields)))

refuse :: String -> [Problem] -> IO a
refuse format problems = do
    Lazy.putStrLn (encode (object ["format" .= format, "status" .= (if exit `elem` [1, 4] then "failed" else "refused" :: String), "means" .= (if exit == 4 then "Execution stopped; the records produced so far are kept and every problem is listed" else "Nothing was produced; every problem is listed" :: String), "does_not_mean" .= ("Anything about the candidate" :: String), "evidence" .= False, "problems" .= map encoded problems]))
    exitWith (ExitFailure exit)
  where
    exit = case maximum (map (severity . code) problems) of 5 -> 1; other -> other
    encoded problem = object [Key.fromString "code" .= code problem, "at" .= at problem, "message" .= message problem]

severity :: String -> Int
severity name
    | name == "internal-error" = 5
    | name `elem` ["identity-mismatch", "execution-failed"] = 4
    | name `elem` ["artifact-missing", "artifact-role", "artifact-invalid"] = 3
    | otherwise = 2

rational :: Rational -> Value
rational value = object ["exact" .= (show (numerator value) ++ "/" ++ show (denominator value)), "decimal" .= (fromRational value :: Double)]
