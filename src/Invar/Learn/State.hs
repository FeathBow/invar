{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.State (Decoder (..), Initial, Observed, observeInitial, observe, initialSchema, initialSteps, initialChecked, steps, checked, compareInitial, compareObserved, compare, compareInitialFiles) where

import Control.Monad (unless)
import Data.Aeson (Value, object, toJSON, withObject, (.:), (.=))
import Data.Aeson.Types (parseEither)
import Data.Map.Strict (Map)
import Data.Text (Text)
import Invar.Learn qualified as Learn
import Invar.Learn.Adapter qualified as Adapter
import Invar.Learn.Checkpoint qualified as Checkpoint
import Invar.Learn.Codec (Decoder (..))
import Invar.Learn.Codec qualified as Codec
import Invar.Learn.Fingerprint qualified as Fingerprint
import Invar.Learn.Observation qualified as Observation
import Invar.Learn.Report qualified as Report
import Invar.Policy.File qualified as File
import System.FilePath ((</>))
import Prelude hiding (compare)

data Initial = Initial Learn.Settings (Map Text [Integer]) [File.Scan] Checkpoint.Checked [Integer] Fingerprint.Fingerprint

data Observed = Observed Report.Report (String, String) [File.Scan] Checkpoint.Checked [Integer] Fingerprint.Fingerprint

observeInitial :: Codec.Session -> (Learn.Settings, FilePath) -> IO Initial
observeInitial session (settings, directory) = do
    (scans, schema) <- File.withFile (directory </> "adapter.safetensors") $ \file -> do
        scans <- Adapter.verifyScan (Learn.policy settings) file
        pure (scans, Adapter.schema file)
    parameters <- either invalid pure (Adapter.parameters schema)
    decoded <- Codec.decode session (directory </> "learner.pt", Learn.learner settings)
    admitted <- either invalid pure (Checkpoint.admitInitial settings parameters decoded)
    counted <- Checkpoint.inspectTensors session admitted
    Initial settings schema scans admitted counted <$> Fingerprint.native session (Checkpoint.value admitted)

initialSchema :: Initial -> Map Text [Integer]
initialSchema (Initial _ schema _ _ _ _) = schema

initialSteps :: Initial -> [Integer]
initialSteps (Initial _ _ _ _ counted _) = counted

initialChecked :: Initial -> Checkpoint.Checked
initialChecked (Initial _ _ _ admitted _ _) = admitted

observe :: Codec.Session -> (Report.Report, Map Text [Integer], FilePath) -> IO Observed
observe session (report, schema, directory) = do
    names@(policy, learner) <- either invalid pure (identities report)
    parameters <- either invalid pure (Adapter.parameters schema)
    scans <- File.withFile (directory </> "adapter.safetensors") $ \file -> Adapter.matches schema file >> Adapter.verifyScan policy file
    decoded <- Codec.decode session (directory </> "learner.pt", learner)
    admitted <- either invalid pure (Checkpoint.admit report parameters decoded)
    counted <- Checkpoint.inspectTensors session admitted
    Observed report names scans admitted counted <$> Fingerprint.native session (Checkpoint.value admitted)

steps :: Observed -> [Integer]
steps (Observed _ _ _ _ counted _) = counted

checked :: Observed -> Checkpoint.Checked
checked (Observed _ _ _ admitted _ _) = admitted

compareInitial :: (Initial, Initial) -> IO Value
compareInitial (Initial leftSettings _ before original _ leftPrint, Initial rightSettings _ after changed _ rightPrint) = do
    learners <- either invalid pure (Fingerprint.differences [toJSON ("learner" :: Text)] (leftPrint, rightPrint))
    let policies = policyChanges (before, after)
    pure (object ["comparison" .= ("initial checkpoint values and tensor bytes" :: Text), "equal" .= (null policies && null learners), "policy_equal" .= null policies, "learner_equal" .= null learners, "left_policy" .= Learn.policy leftSettings, "right_policy" .= Learn.policy rightSettings, "left_learner" .= Learn.learner leftSettings, "right_learner" .= Learn.learner rightSettings, "left_rng" .= object (Checkpoint.rngSummary original), "right_rng" .= object (Checkpoint.rngSummary changed), "differences" .= (policies ++ learners)])

compareObserved :: (Observed, Observed) -> IO Value
compareObserved (Observed first firstIds before _ _ leftPrint, Observed second secondIds after _ _ rightPrint) = do
    learners <- either invalid pure (Fingerprint.differences [toJSON ("learner" :: Text)] (leftPrint, rightPrint))
    summary (first, second) (firstIds, secondIds) (policyChanges (before, after), learners)

compareInitialFiles :: Decoder -> (Learn.Settings, FilePath) -> (Learn.Settings, FilePath) -> IO Value
compareInitialFiles decoder left right = Codec.withSession decoder $ \session -> do
    mapM_ (either (invalid . show) pure . Learn.validate . fst) [left, right]
    before <- observeInitial session left
    after <- observeInitial session right
    compareInitial (before, after)

compare :: Decoder -> FilePath -> (Observation.Input, Observation.Input) -> IO Value
compare decoder policy (left, right) = do
    first <- Observation.report left
    second <- Observation.report right
    either invalid pure (Report.paired first second)
    leftSchema <- inputSchema first policy
    rightSchema <- inputSchema second policy
    Codec.withSession decoder $ \session -> do
        before <- observe session (first, leftSchema, Observation.artifact left)
        after <- observe session (second, rightSchema, Observation.artifact right)
        compareObserved (before, after)

inputSchema :: Report.Report -> FilePath -> IO (Map Text [Integer])
inputSchema report path = do
    expected <- either invalid pure (parseEither (withObject "request" (.: "policy")) (Report.request report))
    File.withFile path (\file -> Adapter.verify expected file >> pure (Adapter.schema file))

policyChanges :: ([File.Scan], [File.Scan]) -> [Value]
policyChanges = Fingerprint.tensorDifferences (\name -> ["path" .= ["policy", name]])

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

summary :: (Report.Report, Report.Report) -> ((String, String), (String, String)) -> ([Value], [Value]) -> IO Value
summary (first, second) ((firstPolicy, firstLearner), (secondPolicy, secondLearner)) (policies, learners) = do
    leftBinding <- binding first
    rightBinding <- binding second
    pure (object ["comparison" .= ("checkpoint values and tensor bytes" :: Text), "equal" .= (null policies && null learners), "policy_equal" .= null policies, "learner_equal" .= null learners, "left_policy" .= firstPolicy, "right_policy" .= secondPolicy, "left_learner" .= firstLearner, "right_learner" .= secondLearner, "left_binding" .= leftBinding, "right_binding" .= rightBinding, "differences" .= (policies ++ learners)])
  where
    binding report = either invalid pure (parseEither (withObject "invocation" (.: "binding")) (Report.invocation report)) :: IO Value

invalid :: String -> IO value
invalid = ioError . userError
