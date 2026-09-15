{-# LANGUAGE OverloadedStrings #-}

module Invar.Replay.Update.Resident (inspect, inspectShared) where

import Control.Monad (foldM, unless, when)
import Data.Aeson (Value (..), object, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8)
import Invar.Infer.Framing qualified as Frame
import Invar.Measurement.Duration qualified as Duration
import Invar.Replay.Update qualified as Update
import Invar.Replay.Update.Artifacts qualified as Artifacts
import Invar.Resident qualified as Boundary
import Invar.Resident.Observation qualified as Resident
import System.FilePath ((</>))
import Text.Printf (printf)

inspect :: ([Update.Update], FilePath) -> (Int, ByteString) -> IO Value
inspect = inspectWith (Boundary.Owner Boundary.Learning 0, "resident")

inspectShared :: ([Update.Update], FilePath) -> (Int, ByteString) -> IO Value
inspectShared = inspectWith (Boundary.Owner Boundary.Shared 0, "shared")

inspectWith :: (Boundary.Owner, Text) -> ([Update.Update], FilePath) -> (Int, ByteString) -> IO Value
inspectWith (owner, mode) (expected, directory) (status, source) = do
    (observations, elapsed) <- either invalid pure (observe owner expected (status, source))
    compared <- traverse compareUpdate (zip [0 :: Int ..] (zip expected observations))
    pure (object ["mode" .= mode, "owner" .= (0 :: Int), "loads" .= (1 :: Int), "updates" .= length expected, "calls" .= compared, "groups" .= map (Resident.describe . snd) observations, "close" .= elapsed])
  where
    compareUpdate (index, (reference, (actual, group))) = do
        let staged = directory </> printf "%04d.checkpoint" index
        fields <- Artifacts.compare (Update.report actual, staged) (Update.report reference, Update.published reference)
        pure (object (["index" .= index, "binding" .= Fields.lookup "binding" (Update.consumed actual), "group" .= Resident.ordinal group] ++ fields))

observe :: Boundary.Owner -> [Update.Update] -> (Int, ByteString) -> Either String ([(Update.Update, Resident.Group)], Duration.Duration)
observe owner expected (status, source) = do
    unless (status == 0) (Left "Resident update replay did not exit successfully")
    when (null expected) (Left "Resident update replay requires at least one update")
    Update.successors expected
    frames <- Frame.decode source
    (current, observations, remaining) <- foldM step (Resident.empty owner, [], frames) expected
    Update.successors (map fst (reverse observations))
    (_, elapsed, rest) <- Resident.finish current remaining
    unless (null rest) (Left "Output follows the resident learning process close")
    pure (reverse observations, elapsed)
  where
    step (current, observations, remaining) reference = do
        actual <- result reference remaining
        unless (Update.consumed actual == Update.consumed reference) (Left "Resident update replay consumed different input")
        (next, group, rest) <- Resident.learningObserved current (Update.report actual) remaining
        pure (next, (actual, group) : observations, rest)

result :: Update.Update -> [Frame.Frame] -> Either String Update.Update
result expected frames = do
    (preceding, output) <- case break (stage "result") frames of
        (before, value : _) -> pure (before, value)
        _ -> Left "Incomplete resident update replay result"
    input <- case filter (stage "consumed") preceding of
        [value] -> pure value
        _ -> Left "Expected one resident update replay consumption"
    Update.decode (object ["consumed_json" .= decodeUtf8 (Frame.raw input), "result_json" .= decodeUtf8 (Frame.raw output), "checkpoint" .= Update.checkpoint expected, "published" .= Update.published expected])
  where
    stage name = (== Just (String name)) . Fields.lookup "stage" . Frame.fields

invalid :: String -> IO value
invalid = ioError . userError
