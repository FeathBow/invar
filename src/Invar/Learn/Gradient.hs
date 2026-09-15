{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Gradient (compare, compareObserved, observe) where

import Control.Monad (unless)
import Data.Aeson (Value, object, withObject, (.:), (.=))
import Data.Aeson.Types (parseEither)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (catMaybes)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Names
import Data.Text.Encoding qualified as Text
import Invar.Json qualified as Json
import Invar.Learn.Adapter qualified as Adapter
import Invar.Learn.Report qualified as Report
import Invar.Policy.File qualified as File
import Invar.Policy.Header qualified as Header
import Prelude hiding (compare)

roles :: [Text]
roles = ["objective", "reward"]

observe :: Map Text [Integer] -> (Report.Report, FilePath) -> IO Value
observe parameters (report, path) = File.withFile path $ \file -> do
    validate report file
    complete (roleInventory parameters) file
    observed <- traverse (\tensor -> (,) (Header.name tensor) <$> File.nonzero file tensor) (File.tensors file)
    let nonzero role = [name | (name, True) <- observed, (role <> "/") `Names.isPrefixOf` name]
    pure (object ["digest" .= Report.gradient report, "tensors" .= length observed, "nonzero_objective_tensors" .= nonzero "objective", "nonzero_reward_tensors" .= nonzero "reward"])

compare :: FilePath -> (Report.Report, FilePath) -> (Report.Report, FilePath) -> IO Value
compare policy (first, left) (second, right) = do
    either invalid pure (Report.paired first second)
    compareObserved (first, policy, left) (second, policy, right)

compareObserved :: (Report.Report, FilePath, FilePath) -> (Report.Report, FilePath, FilePath) -> IO Value
compareObserved (first, leftPolicy, left) (second, rightPolicy, right) = do
    expectedLeft <- File.withFile leftPolicy (inventory first)
    expectedRight <- File.withFile rightPolicy (inventory second)
    File.withFile left $ \initial -> File.withFile right $ \changed -> do
        validate first initial
        validate second changed
        complete expectedLeft initial
        complete expectedRight changed
        differences <- tensorChanges (initial, changed)
        bindings <- traverse (either invalid pure . parseEither (withObject "invocation" (.: "binding")) . Report.invocation) [first, second] :: IO [Value]
        case bindings of
            [leftBinding, rightBinding] -> pure (object ["comparison" .= ("gradient tensor bytes" :: Text), "equal" .= null differences, "left_digest" .= Report.gradient first, "right_digest" .= Report.gradient second, "left_binding" .= leftBinding, "right_binding" .= rightBinding, "left_tensors" .= length (File.tensors initial), "right_tensors" .= length (File.tensors changed), "differences" .= differences])
            _ -> invalid "Missing comparison bindings"

inventory :: Report.Report -> File.File -> IO (Map Text [Integer])
inventory report file = do
    expected <- either invalid pure (parseEither (withObject "request" (.: "policy")) (Report.request report))
    actual <- File.identity file
    unless (actual == expected) (invalid "Input adapter tensor identity differs from consumption")
    entries <- either invalid pure (Adapter.parameters (Adapter.schema file))
    pure (roleInventory entries)

roleInventory :: Map Text [Integer] -> Map Text [Integer]
roleInventory entries = Map.fromList [(role <> "/" <> name, shape) | (name, shape) <- Map.toAscList entries, role <- roles]

validate :: Report.Report -> File.File -> IO ()
validate report file = do
    actual <- File.rawIdentity file
    unless (actual == Report.gradient report) (invalid "Gradient file differs from its reported digest")
    let attributes = File.metadata file
    unless (Map.keys attributes == ["binding", "observation", "policy", "program"]) (invalid "Expected the bound gradient observation metadata")
    bound <- field "binding" attributes >>= either invalid pure . Json.decode . Text.encodeUtf8
    program <- field "program" attributes
    unless (object ["binding" .= bound, "program" .= program] == Report.invocation report) (invalid "Gradient metadata invocation differs from its report")
    policy <- field "policy" attributes
    expected <- either invalid pure (parseEither (withObject "request" (.: "policy")) (Report.request report))
    unless (policy == expected) (invalid "Gradient metadata policy differs from its consumed input")
    observation <- field "observation" attributes
    unless (observation == "objective and reward gradients before AdamW") (invalid "Gradient observation is not the declared pre-AdamW snapshot")
  where
    field name = maybe (invalid "Missing gradient metadata") pure . Map.lookup name

complete :: Map Text [Integer] -> File.File -> IO ()
complete expected file = do
    let actual = Map.fromList [(Header.name tensor, Header.shape tensor) | tensor <- File.tensors file]
    unless (Map.keysSet actual == Map.keysSet expected) (invalid "Gradient parameter inventory mismatch")
    unless (actual == expected) (invalid "Gradient metadata differs from the input adapter")

tensorChanges :: (File.File, File.File) -> IO [Value]
tensorChanges (initial, changed) = catMaybes <$> traverse difference (Set.toAscList (Map.keysSet left `Set.union` Map.keysSet right))
  where
    left = Map.fromList [(Header.name tensor, tensor) | tensor <- File.tensors initial]
    right = Map.fromList [(Header.name tensor, tensor) | tensor <- File.tensors changed]
    difference name = case (Map.lookup name left, Map.lookup name right) of
        (Just before, Just after) -> do
            same <- File.equal (initial, before) (changed, after)
            let fields = ["shape" | Header.shape before /= Header.shape after] ++ ["data" | not same] :: [Text]
            pure (if null fields then Nothing else Just (object ["tensor" .= name, "fields" .= fields]))
        (Nothing, Just tensor) -> File.nonzero changed tensor >> pure (Just (object ["tensor" .= name, "missing" .= ("left" :: Text)]))
        (Just tensor, Nothing) -> File.nonzero initial tensor >> pure (Just (object ["tensor" .= name, "missing" .= ("right" :: Text)]))
        (Nothing, Nothing) -> invalid "Tensor name absent from both inventories"

invalid :: String -> IO value
invalid = ioError . userError
