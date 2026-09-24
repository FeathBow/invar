{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Gradient (Observed, compare, compareObserved, observe) where

import Control.Monad (unless)
import Data.Aeson (Value, object, (.=))
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as Names
import Data.Text.Encoding qualified as Text
import Invar.Json qualified as Json
import Invar.Learn.Adapter qualified as Adapter
import Invar.Learn.Fingerprint qualified as Fingerprint
import Invar.Learn.Report qualified as Report
import Invar.Policy.File qualified as File
import Invar.Policy.Header qualified as Header
import Prelude hiding (compare)

roles :: [Text]
roles = ["objective", "reward"]

data Observed = Observed Report.Report [File.Scan]

observe :: Map Text [Integer] -> (Report.Report, FilePath) -> IO (Value, Observed)
observe parameters (report, path) = File.withFile path $ \file -> do
    (raw, scans) <- File.rawIdentityScan file
    validate report raw file
    complete (roleInventory parameters) file
    let nonzero role = [File.scanName scan | scan <- scans, File.scanNonzero scan, (role <> "/") `Names.isPrefixOf` File.scanName scan]
    pure (object ["digest" .= Report.gradient report, "tensors" .= length scans, "nonzero_objective_tensors" .= nonzero "objective", "nonzero_reward_tensors" .= nonzero "reward"], Observed report scans)

compare :: FilePath -> (Report.Report, FilePath) -> (Report.Report, FilePath) -> IO Value
compare policy (first, left) (second, right) = do
    either invalid pure (Report.paired first second)
    leftParameters <- File.withFile policy (inventory first)
    rightParameters <- File.withFile policy (inventory second)
    (_, before) <- observe leftParameters (first, left)
    (_, after) <- observe rightParameters (second, right)
    compareObserved (before, after)

compareObserved :: (Observed, Observed) -> IO Value
compareObserved (Observed first before, Observed second after) = do
    let differences = Fingerprint.tensorDifferences (\name -> ["tensor" .= name]) (before, after)
    bindings <- traverse (either invalid pure . Report.bindingValue) [first, second] :: IO [Value]
    case bindings of
        [leftBinding, rightBinding] -> pure (object ["comparison" .= ("gradient tensor bytes" :: Text), "equal" .= null differences, "left_digest" .= Report.gradient first, "right_digest" .= Report.gradient second, "left_binding" .= leftBinding, "right_binding" .= rightBinding, "left_tensors" .= length before, "right_tensors" .= length after, "differences" .= differences])
        _ -> invalid "Missing comparison bindings"

inventory :: Report.Report -> File.File -> IO (Map Text [Integer])
inventory report file = do
    expected <- either invalid pure (Report.consumedPolicy report)
    actual <- File.identity file
    unless (actual == expected) (invalid "Input adapter tensor identity differs from consumption")
    either invalid pure (Adapter.parameters (Adapter.schema file))

roleInventory :: Map Text [Integer] -> Map Text [Integer]
roleInventory entries = Map.fromList [(role <> "/" <> name, shape) | (name, shape) <- Map.toAscList entries, role <- roles]

validate :: Report.Report -> String -> File.File -> IO ()
validate report actual file = do
    unless (actual == Report.gradient report) (invalid "Gradient file differs from its reported digest")
    let attributes = File.metadata file
    unless (Map.keys attributes == ["binding", "observation", "policy", "program"]) (invalid "Expected the bound gradient observation metadata")
    bound <- field "binding" attributes >>= either invalid pure . Json.decode . Text.encodeUtf8
    program <- field "program" attributes
    unless (object ["binding" .= bound, "program" .= program] == Report.invocation report) (invalid "Gradient metadata invocation differs from its report")
    policy <- field "policy" attributes
    expected <- either invalid pure (Report.consumedPolicy report)
    unless (Names.unpack policy == expected) (invalid "Gradient metadata policy differs from its consumed input")
    observation <- field "observation" attributes
    unless (observation == "objective and reward gradients before AdamW") (invalid "Gradient observation is not the declared pre-AdamW snapshot")
  where
    field name = maybe (invalid "Missing gradient metadata") pure . Map.lookup name

complete :: Map Text [Integer] -> File.File -> IO ()
complete expected file = do
    let actual = Map.fromList [(Header.name tensor, Header.shape tensor) | tensor <- File.tensors file]
    unless (Map.keysSet actual == Map.keysSet expected) (invalid "Gradient parameter inventory mismatch")
    unless (actual == expected) (invalid "Gradient metadata differs from the input adapter")

invalid :: String -> IO value
invalid = ioError . userError
