{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Changes (compare) where

import Data.Aeson (Value, object, toJSON, (.=))
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Invar.Learn.Codec qualified as Codec
import Invar.Learn.Native qualified as Native
import Prelude hiding (compare)

compare :: Codec.Session -> [Value] -> (Native.Value, Native.Value) -> IO [Value]
compare session path (first, second)
    | Native.typeName first /= Native.typeName second = pure [change path ["type"]]
    | otherwise = values first second
  where
    values left@(Native.Mapping _ _) right@(Native.Mapping _ _) = mappings session path (left, right)
    values (Native.List _ left) (Native.List _ right) = sequences session path (left, right)
    values (Native.Tuple _ left) (Native.Tuple _ right) = sequences session path (left, right)
    values (Native.Tensor left) (Native.Tensor right) = do
        equal <- Codec.equal session (left, right)
        let fields = ["dtype" | Native.dtype left /= Native.dtype right] ++ ["shape" | Native.shape left /= Native.shape right] ++ ["data" | not equal]
        pure [change path fields | not (null fields)]
    values left right = pure [change path ["value"] | left /= right]

mappings :: Codec.Session -> [Value] -> (Native.Value, Native.Value) -> IO [Value]
mappings session path (first, second) = do
    left <- either invalid pure (Native.mapping first)
    right <- either invalid pure (Native.mapping second)
    concat <$> traverse (entry left right) (Set.toAscList (Map.keysSet left `Set.union` Map.keysSet right))
  where
    entry left right key = case (Map.lookup key left, Map.lookup key right) of
        (Just before, Just after) -> compare session (path ++ [toJSON key]) (before, after)
        (Nothing, _) -> pure [object ["path" .= (path ++ [toJSON key]), "missing" .= ("left" :: Text)]]
        (_, Nothing) -> pure [object ["path" .= (path ++ [toJSON key]), "missing" .= ("right" :: Text)]]

sequences :: Codec.Session -> [Value] -> ([Native.Value], [Native.Value]) -> IO [Value]
sequences session path (first, second)
    | length first /= length second = pure [change path ["length"]]
    | otherwise = concat <$> traverse (\(index, pair) -> compare session (path ++ [toJSON index]) pair) (zip [0 :: Integer ..] (zip first second))

change :: [Value] -> [Text] -> Value
change path fields = object ["path" .= path, "fields" .= fields]

invalid :: String -> IO value
invalid = ioError . userError
