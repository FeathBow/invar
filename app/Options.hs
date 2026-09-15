module Options (Fields, descriptions, prefixed, parse, required, optional, numeric) where

import Control.Monad (unless)
import Data.Bifunctor (first)
import Data.Map.Strict qualified as Map
import System.Console.GetOpt (ArgDescr (ReqArg), ArgOrder (Permute), OptDescr (Option), getOpt)
import Text.Read (readMaybe)

type Fields = Map.Map String String

descriptions :: [(String, String)] -> [OptDescr (String, String)]
descriptions = map (\(name, description) -> Option [] [name] (ReqArg (name,) "VALUE") description)

prefixed :: String -> [OptDescr (String, String)] -> [OptDescr (String, String)]
prefixed prefix = map rename
  where
    rename (Option _ names argument description) = Option [] (map (prefix ++) names) (fmap (first (prefix ++)) argument) description

parse :: [OptDescr (String, String)] -> [String] -> Either String Fields
parse options supplied = do
    let (pairs, positional, errors) = getOpt Permute options supplied
        fields = Map.fromList pairs
    unless (null errors && null positional) (Left (concat errors ++ "Unexpected command arguments"))
    unless (length pairs == Map.size fields) (Left "Duplicate options are not allowed")
    pure fields

required :: Fields -> String -> Either String String
required fields name = maybe (Left ("Missing option: --" ++ name)) Right (Map.lookup name fields)

optional :: Fields -> String -> Maybe String
optional fields name = Map.lookup name fields

numeric :: (Read value) => Fields -> String -> Either String value
numeric fields name = do
    value <- required fields name
    maybe (Left ("Invalid number for --" ++ name)) Right (readMaybe value)
