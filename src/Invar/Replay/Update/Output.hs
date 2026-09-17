{-# LANGUAGE OverloadedStrings #-}

module Invar.Replay.Update.Output (Report, inspect, inspectResident, inspectShared, describe) where

import Control.Monad (unless)
import Data.Aeson (Object, ToJSON (..), Value (..), object, withObject, (.=))
import Data.Aeson.Encoding qualified as Encoding
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Pair, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8)
import Invar.Json qualified as Json
import Invar.Learn.Report qualified as Learning
import Invar.Measurement.Duration qualified as Duration
import Invar.Replay.Update qualified as Update
import Invar.Replay.Update.Artifacts qualified as Artifacts
import Invar.Replay.Update.Resident qualified as Resident

data Report = Report (Map Text Duration.Duration) [Pair]
data Frame = Frame ByteString Object

inspectResident :: ([Update.Update], FilePath) -> (Int, ByteString) -> IO Value
inspectResident = Resident.inspect

inspectShared :: ([Update.Update], FilePath) -> (Int, ByteString) -> IO Value
inspectShared = Resident.inspectShared

inspect :: (Update.Update, FilePath) -> (Int, ByteString) -> IO Report
inspect (expected, staged) (status, encoded) = do
    (actual, measured) <- either invalid pure (observe expected status encoded)
    compared <- Artifacts.compare (actual, staged) (Update.report expected, Update.published expected)
    pure (Report measured compared)

observe :: Update.Update -> Int -> ByteString -> Either String (Learning.Report, Map Text Duration.Duration)
observe expected status encoded = do
    unless (status == 0) (Left "Replayed update process did not exit successfully")
    unless ("\n" `Bytes.isSuffixOf` encoded) (Left "Incomplete replayed update stream")
    frames <- traverse frame (Bytes.lines encoded)
    let allowed = ["loading", "profile", "load", "loaded_learner", "probability_roles", "roles", "consumed", "reward_update", "artifacts", "checkpoint", "result"]
    unless (all (\(Frame _ fields) -> not (Fields.member "phase" fields) && stage fields `elem` map Just allowed) frames) (Left "Unknown replayed update stage")
    input <- exactly "consumed" frames
    output <- exactly "result" frames
    let Frame first consumed = input
        Frame final _ = output
    unless (consumed == Update.consumed expected) (Left "Replayed consumption differs from the reference")
    actual <- Update.decode (object ["consumed_json" .= decodeUtf8 first, "result_json" .= decodeUtf8 final, "checkpoint" .= Update.checkpoint expected, "published" .= Update.published expected])
    durations <- traverse (duration frames) ["load", "probability_roles", "reward_update"]
    case durations of
        (_, initial) : remaining -> unless (all (Duration.sameClock initial . snd) remaining) (Left "Mixed replayed update measurement clocks")
        [] -> Left "Missing replayed update measurements"
    let ordering = [name | Frame _ fields <- frames, Just name <- [stage fields], name `elem` ["load", "probability_roles", "consumed", "reward_update", "result"]]
    unless (ordering == ["load", "probability_roles", "consumed", "reward_update", "result"] && endsAtResult frames) (Left "Replayed update stages are out of order or incomplete")
    pure (Update.report actual, Map.fromList durations)
  where
    duration frames name = do
        Frame raw fields <- exactly name frames
        (,) name <$> Duration.admit raw fields
    endsAtResult frames = case reverse frames of
        Frame _ fields : _ -> stage fields == Just "result"
        [] -> False

frame :: ByteString -> Either String Frame
frame raw = Frame raw <$> (Json.decode raw >>= parseEither (withObject "update replay record" pure))

stage :: Object -> Maybe Text
stage fields = case Fields.lookup "stage" fields of
    Just (String name) -> Just name
    _ -> Nothing

exactly :: Text -> [Frame] -> Either String Frame
exactly name frames = case [event | event@(Frame _ fields) <- frames, stage fields == Just name] of
    [event] -> pure event
    _ -> Left ("Expected one replayed " ++ show name ++ " record")

describe :: Report -> Value
describe (Report measured compared) = object (compared ++ ["measured" .= measured])

instance ToJSON Report where
    toJSON = describe
    toEncoding (Report measured compared) = Encoding.pairs (foldMap (uncurry (.=)) compared <> "measured" .= measured)

invalid :: String -> IO value
invalid = ioError . userError
