{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Adapter (schema, parameters, mlxParameters, verify, matches) where

import Control.Monad (unless)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as Text
import Invar.Policy.File qualified as File
import Invar.Policy.Header qualified as Header

schema :: File.File -> Map Text [Integer]
schema = Map.fromList . map (\tensor -> (Header.name tensor, Header.shape tensor)) . File.tensors

parameters :: Map Text [Integer] -> Either String (Map Text [Integer])
parameters entries
    | mlxParameters entries = Right entries
    | otherwise = Map.fromList <$> traverse parameter (Map.toAscList entries)
  where
    parameter (name, shape) = do
        unless (any (`Text.isSuffixOf` name) [".lora_A.weight", ".lora_B.weight"]) (Left "Unsupported adapter parameter in the default LoRA profile")
        case Text.stripSuffix ".weight" name of
            Just stem -> pure (stem <> ".default.weight", shape)
            Nothing -> Left "Missing LoRA weight suffix"

mlxParameters :: Map Text [Integer] -> Bool
mlxParameters entries = not (Map.null entries) && all native (Map.keys entries)
  where
    native name = any (`Text.isSuffixOf` name) [".lora_a", ".lora_b"]

verify :: String -> File.File -> IO ()
verify expected file = do
    actual <- File.identity file
    unless (actual == expected) (ioError (userError "Adapter contents do not match the requested tensor identity"))

matches :: Map Text [Integer] -> File.File -> IO ()
matches expected file = do
    let actual = schema file
    unless (Map.keysSet actual == Map.keysSet expected) (ioError (userError "Adapter keys do not match the resolved model targets"))
    unless (actual == expected) (ioError (userError "Adapter tensor metadata mismatch"))
