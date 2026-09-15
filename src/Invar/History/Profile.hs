{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Profile (Profiles (..), Observation, fromPrefix, final, summarize) where

import Control.Monad (unless)
import Data.Aeson (Object, Value (..), object, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Maybe (catMaybes, listToMaybe)
import Data.Text (Text)
import Invar.Resident qualified as Resident
import Numeric.Natural (Natural)

data Profiles = Unreported | Uniform | Roles deriving (Eq, Show)
data Observation = Observation Resident.Role (Maybe Value)

-- Execution framing has already admitted this actual loading prefix.
fromPrefix :: Resident.Role -> [Object] -> [Observation]
fromPrefix role records = [Observation role reported | fields <- records, stage fields == Just "load"]
  where
    reported = Object <$> listToMaybe [fields | fields <- records, stage fields == Just "profile"]

final :: Profiles -> [Value] -> Either String [Observation]
final mode records = do
    let prefix = [fields | Object fields <- records, stage fields `elem` map Just ["loading", "profile", "load"]]
        expected = if mode == Unreported then ["load"] else ["loading", "profile", "load"]
    unless (map stage prefix == map Just expected) (Left "Final inference loading records differ from the declared profile mode")
    pure (fromPrefix Resident.Inference prefix)

summarize :: Profiles -> [Observation] -> Either String Value
summarize Unreported observations = do
    unless (null [value | Observation _ (Just value) <- observations]) (Left "History profiles differ from the declared profile mode")
    pure (object ["mode" .= ("unreported" :: Text), "count" .= (0 :: Natural)])
summarize Uniform observations = do
    (count, profile) <- uniform observations
    pure (object ["mode" .= ("uniform" :: Text), "count" .= count, "profile" .= profile])
summarize Roles observations = do
    inference <- selected Resident.Inference
    learning <- selected Resident.Learning
    shared <- case [observed | observed@(Observation actual _) <- observations, actual == Resident.Shared] of
        [] -> pure Nothing
        values -> Just <$> uniform values
    pure
        ( object $
            [ "mode" .= ("roles" :: Text)
            , "count" .= (fst inference + fst learning + maybe 0 fst shared)
            , "inference" .= describe inference
            , "learning" .= describe learning
            ]
                ++ maybe [] (\value -> ["shared" .= describe value]) shared
        )
  where
    selected role = uniform [observed | observed@(Observation actual _) <- observations, actual == role]
    describe (count, profile) = object ["count" .= count, "profile" .= profile]

uniform :: [Observation] -> Either String (Natural, Value)
uniform observations = case catMaybes [profile | Observation _ profile <- observations] of
    first : rest -> do
        unless (length observations == length rest + 1 && all (== first) rest) (Left "History profile count or complete payloads differ across declared processes")
        pure (fromIntegral (length observations), first)
    [] -> Left "History profiles differ from the declared profile mode"

stage :: Object -> Maybe Text
stage fields = case Fields.lookup "stage" fields of
    Just (String value) -> Just value
    _ -> Nothing
