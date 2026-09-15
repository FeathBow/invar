{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.State (Decoder (..), compare, compareInitial, compareObserved) where

import Control.Monad (unless)
import Data.Aeson (Value, object, toJSON, withObject, (.:), (.=))
import Data.Aeson.Types (parseEither)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (catMaybes)
import Data.Set qualified as Set
import Data.Text (Text)
import Invar.Learn qualified as Learn
import Invar.Learn.Adapter qualified as Adapter
import Invar.Learn.Changes qualified as Changes
import Invar.Learn.Checkpoint qualified as Checkpoint
import Invar.Learn.Codec (Decoder (..))
import Invar.Learn.Codec qualified as Codec
import Invar.Learn.Native qualified as Native
import Invar.Learn.Observation qualified as Observation
import Invar.Learn.Report qualified as Report
import Invar.Policy.File qualified as File
import Invar.Policy.Header qualified as Header
import System.FilePath ((</>))
import Prelude hiding (compare)

compareInitial :: Decoder -> (Learn.Settings, FilePath) -> (Learn.Settings, FilePath) -> IO Value
compareInitial decoder left@(leftSettings, leftPath) right@(rightSettings, rightPath) = do
    mapM_ (either (invalid . show) pure . Learn.validate) [leftSettings, rightSettings]
    File.withFile (leftPath </> "adapter.safetensors") $ \before ->
        File.withFile (rightPath </> "adapter.safetensors") $ \after -> do
            Adapter.verify (Learn.policy leftSettings) before
            Adapter.verify (Learn.policy rightSettings) after
            leftParameters <- either invalid pure (Adapter.parameters (Adapter.schema before))
            rightParameters <- either invalid pure (Adapter.parameters (Adapter.schema after))
            Codec.withSession decoder $ \session -> do
                original <- initialCheckpoint (session, leftParameters) left
                changed <- initialCheckpoint (session, rightParameters) right
                let originalValue = Checkpoint.value original
                    changedValue = Checkpoint.value changed
                distinctReferences [originalValue, changedValue]
                policies <- policyChanges (before, after)
                learners <- Changes.compare session [toJSON ("learner" :: Text)] (originalValue, changedValue)
                pure (object ["comparison" .= ("initial checkpoint values and tensor bytes" :: Text), "equal" .= (null policies && null learners), "policy_equal" .= null policies, "learner_equal" .= null learners, "left_policy" .= Learn.policy leftSettings, "right_policy" .= Learn.policy rightSettings, "left_learner" .= Learn.learner leftSettings, "right_learner" .= Learn.learner rightSettings, "left_rng" .= object (Checkpoint.rngSummary original), "right_rng" .= object (Checkpoint.rngSummary changed), "differences" .= (policies ++ learners), "scope" .= ("initial checkpoint contents; not execution, restoration or numerical qualification" :: Text)])

initialCheckpoint :: (Codec.Session, Map Text [Integer]) -> (Learn.Settings, FilePath) -> IO Checkpoint.Checked
initialCheckpoint (session, parameters) (settings, directory) = do
    decoded <- Codec.decode session (directory </> "learner.pt", Learn.learner settings)
    checked <- either invalid pure (Checkpoint.admitInitial settings parameters decoded)
    _ <- Checkpoint.inspectTensors session checked
    pure checked

distinctReferences :: [Native.Value] -> IO ()
distinctReferences values = do
    let references = map Native.index (concatMap Native.tensorValues values)
    unless (length references == Set.size (Set.fromList references)) (invalid "Native tensor references were reused across checkpoint snapshots")

compare :: Decoder -> FilePath -> (Observation.Input, Observation.Input) -> IO Value
compare decoder policy (left, right) = do
    first <- Observation.report left
    second <- Observation.report right
    either invalid pure (Report.paired first second)
    compareObserved decoder (first, policy, Observation.artifact left) (second, policy, Observation.artifact right)

-- Each side supplies its own consumed policy and output checkpoint. Numerical
-- input equality is a separate observation in a complete history comparison.
compareObserved :: Decoder -> (Report.Report, FilePath, FilePath) -> (Report.Report, FilePath, FilePath) -> IO Value
compareObserved decoder (first, leftPolicy, left) (second, rightPolicy, right) = do
    leftSchema <- inputSchema first leftPolicy
    rightSchema <- inputSchema second rightPolicy
    leftParameters <- either invalid pure (Adapter.parameters leftSchema)
    rightParameters <- either invalid pure (Adapter.parameters rightSchema)
    firstIds <- either invalid pure (identities first)
    secondIds <- either invalid pure (identities second)
    File.withFile (left </> "adapter.safetensors") $ \before ->
        File.withFile (right </> "adapter.safetensors") $ \after -> do
            Adapter.verify (fst firstIds) before
            Adapter.verify (fst secondIds) after
            Adapter.matches leftSchema before
            Adapter.matches rightSchema after
            Codec.withSession decoder $ \session -> do
                original <- checkpoint (session, first, leftParameters) (left, snd firstIds)
                changed <- checkpoint (session, second, rightParameters) (right, snd secondIds)
                distinctReferences [original, changed]
                policies <- policyChanges (before, after)
                learners <- Changes.compare session [toJSON ("learner" :: Text)] (original, changed)
                summary (first, second) (firstIds, secondIds) (policies, learners)

inputSchema :: Report.Report -> FilePath -> IO (Map Text [Integer])
inputSchema report path = do
    expected <- either invalid pure (parseEither (withObject "request" (.: "policy")) (Report.request report))
    File.withFile path (\file -> Adapter.verify expected file >> pure (Adapter.schema file))

identities :: Report.Report -> Either String (String, String)
identities report = do
    policy <- Report.artifact "adapter" report
    learner <- Report.artifact "learner" report
    expected <- parseEither (withObject "request" (.: "policy")) (Report.request report)
    parseEither (withObject "staged update result" (check policy expected)) (Report.result report)
    pure (policy, learner)
  where
    check policy expected fields = do
        storage <- fields .: "storage"
        unless (storage == ("staged; not published" :: Text)) (fail "Expected a staged update report")
        update <- fields .: "update"
        before <- update .: "before"
        after <- update .: "after"
        unless (before == (expected :: String) && after == policy) (fail "Update policy identities disagree")

checkpoint :: (Codec.Session, Report.Report, Map Text [Integer]) -> (FilePath, String) -> IO Native.Value
checkpoint (session, report, parameters) (directory, digest) = do
    decoded <- Codec.decode session (directory </> "learner.pt", digest)
    checked <- either invalid pure (Checkpoint.admit report parameters decoded)
    _ <- Checkpoint.inspectTensors session checked
    pure (Checkpoint.value checked)

policyChanges :: (File.File, File.File) -> IO [Value]
policyChanges (first, second) = catMaybes <$> traverse difference (Set.toAscList (Map.keysSet left `Set.union` Map.keysSet right))
  where
    left = Map.fromList [(Header.name tensor, tensor) | tensor <- File.tensors first]
    right = Map.fromList [(Header.name tensor, tensor) | tensor <- File.tensors second]
    difference name = case (Map.lookup name left, Map.lookup name right) of
        (Just before, Just after) -> do
            same <- File.equal (first, before) (second, after)
            let fields = ["shape" | Header.shape before /= Header.shape after] ++ ["data" | not same] :: [Text]
            pure (if null fields then Nothing else Just (object ["path" .= ["policy", name], "fields" .= fields]))
        (Nothing, _) -> pure (Just (object ["path" .= ["policy", name], "missing" .= ("left" :: Text)]))
        (_, Nothing) -> pure (Just (object ["path" .= ["policy", name], "missing" .= ("right" :: Text)]))

summary :: (Report.Report, Report.Report) -> ((String, String), (String, String)) -> ([Value], [Value]) -> IO Value
summary (first, second) ((firstPolicy, firstLearner), (secondPolicy, secondLearner)) (policies, learners) = do
    leftBinding <- binding first
    rightBinding <- binding second
    pure (object ["comparison" .= ("checkpoint values and tensor bytes" :: Text), "equal" .= (null policies && null learners), "policy_equal" .= null policies, "learner_equal" .= null learners, "left_policy" .= firstPolicy, "right_policy" .= secondPolicy, "left_learner" .= firstLearner, "right_learner" .= secondLearner, "left_binding" .= leftBinding, "right_binding" .= rightBinding, "differences" .= (policies ++ learners), "scope" .= ("reported checkpoint contents; not execution, restoration or numerical qualification" :: Text)])
  where
    binding report = either invalid pure (parseEither (withObject "invocation" (.: "binding")) (Report.invocation report)) :: IO Value

invalid :: String -> IO value
invalid = ioError . userError
