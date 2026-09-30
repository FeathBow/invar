{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Trace (Run (..), Mode (..), Checked, Generation, admit, describe, generations, cohort, publication, diagnostics, closing, modelLoads, stepOutputs, profiles) where

import Control.Monad (foldM, unless)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Object, Value (..), object, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text (Text)
import Invar.Artifact qualified as Artifact
import Invar.History.Cohort qualified as Cohort
import Invar.History.Execution (Frame (..), Mode (..))
import Invar.History.Execution qualified as Execution
import Invar.History.Profile qualified as Profile
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Learn qualified as Learn
import Invar.Learn.Report qualified as Report
import Invar.Resident.Observation qualified as Resident
import Invar.Spec.Invocation qualified as V
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)
import System.FilePath ((</>))

data Run = Run {settings :: Learn.Settings, sessions :: Natural, output :: FilePath, method :: String, exitCode :: Int, inferenceMode :: Mode, learningMode :: Mode}

data Checked = Checked String [Generation] [Frame]
data Generation = Generation Cohort.Checked Object Object [Frame] [Resident.Group] [Profile.Observation]

admit :: Run -> Workload.Document -> ByteString -> Either String Checked
admit run declared encoded = do
    first show (Learn.validate (settings run))
    unless (exitCode run == 0) (Left "Training process did not exit successfully")
    unless (sessions run > 0 && method run `elem` ["rename", "reference"] && not (null (output run))) (Left "Invalid declared training sessions, publication method or output directory")
    unless ("\n" `Bytes.isSuffixOf` encoded) (Left "Incomplete final training observation line")
    frames <- traverse frame (Bytes.lines encoded)
    initial <- Execution.start (inferenceMode run, learningMode run) (sessions run)
    (_, _, accepted, state, remaining) <- foldM (advance run) (settings run, 0, [], initial, frames) (zip [0 ..] (Workload.cycles declared))
    closed <- Execution.finish state remaining
    pure (Checked (Artifact.hex (SHA256.hash encoded)) (reverse accepted) closed)

frame :: ByteString -> Either String Frame
frame encoded = Frame encoded <$> (Json.decode encoded >>= parseEither (withObject "training log record" pure))

advance :: Run -> (Learn.Settings, Natural, [Generation], Execution.State, [Frame]) -> (Natural, Workload.Cycle) -> Either String (Learn.Settings, Natural, [Generation], Execution.State, [Frame])
advance run (current, offset, accepted, state, remaining) (index, workload) = do
    let (execution, ending) = break (phase "cycle") remaining
        count = fromIntegral (length (Workload.tasks workload))
        call = offset + count
    (finished, rest) <- case ending of
        Frame _ fields : rest -> Right (fields, rest)
        [] -> Left "Missing declared training cycle completion"
    parseEither (cycleSummary (index, sessions run)) finished
    (body, published) <- case reverse execution of
        Frame _ fields : reversed | Fields.lookup "phase" fields == Just (String "published") -> Right (reverse reversed, fields)
        _ -> Left "Missing publication immediately before cycle completion"
    observed <- Cohort.admitFrames current (workload, call) [Framing.Frame raw fields | Frame raw fields <- body]
    updated <- parseEither (withObject "update invocation" Wire.binding) (Report.invocation (Cohort.update observed))
    unless (updated == binding call) (Left "Update invocation identity differs from the declared history")
    (next, groups, modelProfiles) <- Execution.validate (current, state) (workload, offset, observed) body
    parseEither (publish (run, index, offset) (workload, observed)) published
    policy <- Report.artifact "adapter" (Cohort.update observed)
    learner <- Report.artifact "learner" (Cohort.update observed)
    let diagnostic = [observedFrame | observedFrame@(Frame _ fields) <- body, Fields.lookup "stage" fields `elem` map (Just . String) ["loading", "profile", "load", "activation", "released", "loaded_learner", "inference", "proximal", "current", "applied", "reward_update", "artifacts", "checkpoint"]]
    pure (current {Learn.policy, Learn.learner, Learn.schedule = Learn.synchronous (index + 1) policy}, call + 1, Generation observed published finished diagnostic groups modelProfiles : accepted, next, rest)

phase :: Text -> Frame -> Bool
phase expected (Frame _ fields) = Fields.lookup "phase" fields == Just (String expected)

cycleSummary :: (Natural, Natural) -> Object -> Parser ()
cycleSummary (index, count) fields = do
    Json.fields ["phase", "index", "sessions", "seconds"] fields
    actualIndex <- fields .: "index"
    actualCount <- fields .: "sessions"
    seconds <- fields .: "seconds" >>= Json.finite
    unless (actualIndex == index && actualCount == count && seconds >= 0) (fail "Training cycle index, sessions or timing differs from its declaration")

publish :: (Run, Natural, Natural) -> (Workload.Cycle, Cohort.Checked) -> Object -> Parser ()
publish (run, index, offset) (workload, observed) fields = do
    Json.fields ["phase", "checkpoint", "policy", "learner", "publication", "binding", "delivery"] fields
    let reported = Cohort.update observed
    policy <- either fail pure (Report.artifact "adapter" reported)
    learner <- either fail pure (Report.artifact "learner" reported)
    updated <- withObject "update invocation" Wire.binding (Report.invocation reported)
    actualPath <- fields .: "checkpoint"
    actualMethod <- fields .: "publication"
    actualPolicy <- fields .: "policy"
    actualLearner <- fields .: "learner"
    actualBinding <- fields .: "binding"
    actualDelivery <- fields .: "delivery"
    unless (actualPath == output run </> ("generation" ++ show (index + 1)) && actualMethod == method run && actualPolicy == policy && actualLearner == learner && actualBinding == Wire.bindingValue updated) (fail "Training publication differs from the declared output or update")
    unless (actualDelivery == map (Wire.bindingValue . binding . (offset +)) (Workload.delivery workload)) (fail "Training delivery differs from its declared cohort")

binding :: Natural -> V.Binding
binding = V.ordinal

generations :: Checked -> [Generation]
generations (Checked _ values _) = values

cohort :: Generation -> Cohort.Checked
cohort (Generation value _ _ _ _ _) = value

publication :: Generation -> Value
publication (Generation _ value _ _ _ _) = Object value

diagnostics :: Generation -> [Value]
diagnostics (Generation _ _ _ values _ _) = [Object fields | Frame _ fields <- values]

stepOutputs :: Generation -> [ByteString]
stepOutputs (Generation _ _ _ values _ _) = [encoded | Frame encoded fields <- values, Fields.lookup "stage" fields `elem` map (Just . String) ["proximal", "current"]]

profiles :: Generation -> [Profile.Observation]
profiles (Generation _ _ _ _ _ values) = values

describe :: Checked -> Value
describe (Checked digest accepted closed) = object ["log_sha256" .= digest, "generations" .= map generation accepted, "closed" .= [Object fields | Frame _ fields <- closed]]
  where
    generation (Generation observed published finished diagnostic groups _) = object ["cohort" .= Cohort.describe observed, "publication" .= Object published, "cycle" .= Object finished, "diagnostics" .= [Object fields | Frame _ fields <- diagnostic], "resident_groups" .= map Resident.describe groups]

closing :: Checked -> [Value]
closing (Checked _ _ values) = [Object fields | Frame _ fields <- values]

modelLoads :: Generation -> Natural
modelLoads (Generation _ _ _ records _ _) = Execution.modelLoads records
