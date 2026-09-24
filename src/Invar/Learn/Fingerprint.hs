{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Fingerprint (Fingerprint, native, disjoint, differences, tensorDifferences) where

import Control.Monad (unless)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value, object, toJSON, (.=))
import Data.Aeson.Types (Pair)
import Data.IORef (modifyIORef', newIORef, readIORef)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (mapMaybe)
import Data.Set qualified as Set
import Data.Text (Text)
import Invar.Artifact qualified as Artifact
import Invar.Learn.Codec qualified as Codec
import Invar.Learn.Native qualified as Native
import Invar.Policy.File qualified as File
import Numeric.Natural (Natural)

data Fingerprint = Fingerprint Native.Value (Map Natural String)
    deriving (Eq, Show)

native :: Codec.Session -> Native.Value -> IO Fingerprint
native session value = do
    let found = Native.tensorValues value
        references = map Native.index found
    unless (length references == Set.size (Set.fromList references)) (ioError (userError "Native tensor references were reused within one checkpoint"))
    Fingerprint value . Map.fromList <$> traverse digest found
  where
    digest tensor = do
        context <- newIORef SHA256.init
        Codec.consume (session, tensor) (\chunk -> modifyIORef' context (`SHA256.update` chunk))
        (,) (Native.index tensor) . Artifact.hex . SHA256.finalize <$> readIORef context

disjoint :: [Fingerprint] -> Either String ()
disjoint prints = unless (length references == Set.size (Set.fromList references)) (Left "Native tensor references were reused across checkpoint snapshots")
  where
    references = concat [Map.keys digests | Fingerprint _ digests <- prints]

differences :: [Value] -> (Fingerprint, Fingerprint) -> Either String [Value]
differences path (Fingerprint first left, Fingerprint second right) = walk path (first, second)
  where
    walk at (before, after)
        | Native.typeName before /= Native.typeName after = Right [change at ["type"]]
        | otherwise = values at before after
    values at before@(Native.Mapping _ _) after@(Native.Mapping _ _) = do
        leftItems <- Native.mapping before
        rightItems <- Native.mapping after
        concat <$> traverse (entry at leftItems rightItems) (Set.toAscList (Map.keysSet leftItems `Set.union` Map.keysSet rightItems))
    values at (Native.List _ before) (Native.List _ after) = sequences at (before, after)
    values at (Native.Tuple _ before) (Native.Tuple _ after) = sequences at (before, after)
    values at (Native.Tensor before) (Native.Tensor after) =
        let fields = ["dtype" | Native.dtype before /= Native.dtype after] ++ ["shape" | Native.shape before /= Native.shape after] ++ ["data" | Map.lookup (Native.index before) left /= Map.lookup (Native.index after) right]
         in Right [change at fields | not (null fields)]
    values at before after = Right [change at ["value"] | before /= after]
    entry at leftItems rightItems key = case (Map.lookup key leftItems, Map.lookup key rightItems) of
        (Just before, Just after) -> walk (at ++ [toJSON key]) (before, after)
        (Nothing, _) -> Right [object ["path" .= (at ++ [toJSON key]), "missing" .= ("left" :: Text)]]
        (_, Nothing) -> Right [object ["path" .= (at ++ [toJSON key]), "missing" .= ("right" :: Text)]]
    sequences at (before, after)
        | length before /= length after = Right [change at ["length"]]
        | otherwise = concat <$> traverse (\(index, pair) -> walk (at ++ [toJSON index]) pair) (zip [0 :: Integer ..] (zip before after))
    change at fields = object ["path" .= at, "fields" .= (fields :: [Text])]

tensorDifferences :: (Text -> [Pair]) -> ([File.Scan], [File.Scan]) -> [Value]
tensorDifferences label (first, second) = mapMaybe difference (Set.toAscList (Map.keysSet left `Set.union` Map.keysSet right))
  where
    left = Map.fromList [(File.scanName scan, scan) | scan <- first]
    right = Map.fromList [(File.scanName scan, scan) | scan <- second]
    difference name = case (Map.lookup name left, Map.lookup name right) of
        (Just before, Just after) ->
            let fields = ["shape" | File.scanShape before /= File.scanShape after] ++ ["data" | File.scanDigest before /= File.scanDigest after] :: [Text]
             in if null fields then Nothing else Just (object (label name ++ ["fields" .= fields]))
        (Nothing, _) -> Just (object (label name ++ ["missing" .= ("left" :: Text)]))
        (_, Nothing) -> Just (object (label name ++ ["missing" .= ("right" :: Text)]))
