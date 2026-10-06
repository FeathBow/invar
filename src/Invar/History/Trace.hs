{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Trace (Run (..), Mode (..), Checked, Generation, admit, describe, generations, finalPolicy, cohort, publication, diagnostics, closing, modelLoads, stepOutputs, profiles) where

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
import Invar.Cohort qualified as C
import Invar.History.Cohort qualified as Cohort
import Invar.History.Execution (Frame (..), Mode (..))
import Invar.History.Execution qualified as Execution
import Invar.History.Profile qualified as Profile
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Learn qualified as Learn
import Invar.Learn.Observed qualified as Input
import Invar.Learn.Report qualified as Report
import Invar.Learn.Step qualified as Step
import Invar.Policy qualified as Policy
import Invar.Resident.Group qualified as Group
import Invar.Spec.Invocation qualified as V
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)
import System.FilePath ((</>))

data Run = Run {settings :: Learn.Settings, sessions :: Natural, output :: FilePath, method :: String, exitCode :: Int, inferenceMode :: Mode, learningMode :: Mode}

data Checked = Checked String [Generation] [Frame] Policy.Description
data Generation = Generation Cohort.Checked Object Object [Frame] [Group.Group] [Profile.Observation]
data Waiting = Waiting Learn.Settings [C.Task] Natural Execution.Cycle Report.Report Object Object [Frame]

admit :: Run -> Policy.Description -> Workload.Document -> ByteString -> Either String Checked
admit run initial declared encoded = do
    first show (Learn.validate (settings run))
    unless (exitCode run == 0) (Left "Training process did not exit successfully")
    unless (sessions run > 0 && method run `elem` ["rename", "reference"] && not (null (output run))) (Left "Invalid declared training sessions, publication method or output directory")
    let chosen = settings run
    unless (Policy.bindings initial == (Learn.policy chosen, Learn.tokenizer chosen, Learn.behaviorBase chosen, Learn.behaviorAssembly chosen)) (Left "Initial policy description differs from the declared inference materialization")
    unless ("\n" `Bytes.isSuffixOf` encoded) (Left "Incomplete final training observation line")
    frames <- traverse frame (Bytes.lines encoded)
    started <- Execution.start (inferenceMode run, learningMode run) (sessions run)
    (_, final, _, waiting, state, remaining) <- foldM (advance run) (chosen, initial, 0, [], started, frames) (zip [0 ..] (Workload.cycles declared))
    closed <- Execution.finish state remaining
    accepted <- traverse admitted (reverse waiting)
    pure (Checked (Artifact.hex (SHA256.hash encoded)) accepted closed final)
  where
    admitted (Waiting current tasks offset executed reported published finished diagnostic) = do
        delimited <- first show (concat <$> traverse Replay.delimit (Execution.pending executed))
        let logged = delimited ++ Execution.admitted executed
            matching position = [selected | selected <- logged, Trajectory.binding (Replay.trajectory selected) == binding (offset + position)]
        ordered <- traverse (\position -> case matching position of [single] -> Right single; _ -> Left "Expected one admitted inference for each declared cohort member") [0 .. fromIntegral (length tasks) - 1]
        observed <- Cohort.admit current (tasks, ordered) reported
        pure (Generation observed published finished diagnostic (Execution.groups executed) (Execution.profiles executed))

frame :: ByteString -> Either String Frame
frame encoded = Frame encoded <$> (Json.decode encoded >>= parseEither (withObject "training log record" pure))

advance :: Run -> (Learn.Settings, Policy.Description, Natural, [Waiting], Execution.State, [Frame]) -> (Natural, Workload.Cycle) -> Either String (Learn.Settings, Policy.Description, Natural, [Waiting], Execution.State, [Frame])
advance run (current, described, offset, accepted, state, remaining) (index, workload) = do
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
    reported <- Report.admitFrames call [Framing.Frame raw fields | Frame raw fields <- body]
    updated <- parseEither (withObject "update invocation" Wire.binding) (Report.invocation reported)
    unless (updated == binding call) (Left "Update invocation identity differs from the declared history")
    tasks <- traverse (Input.task current (Just described)) (Workload.tasks workload)
    calls <- traverse (\(position, task) -> first show (Call.prepare (binding (offset + position)) (C.plan task))) (zip [0 ..] tasks)
    let scoring = if Learn.reference current == Learn.policy current then Nothing else Just (Learn.reference current)
    (next, executed) <- Execution.validate (current, state) (workload, Execution.Declared calls scoring reported) body
    parseEither (publish (run, index, offset) (workload, reported)) published
    policy <- Report.artifact "adapter" reported
    learner <- Report.artifact "learner" reported
    successor <- Policy.successor policy described
    let diagnostic = [observedFrame | observedFrame@(Frame _ fields) <- body, Fields.lookup "stage" fields `elem` map (Just . String) (["loading", "profile", "load", "activation", "released", "loaded_learner", "inference"] ++ Step.stages ++ ["reward_update", "artifacts", "checkpoint"])]
    pure (current {Learn.policy, Learn.learner, Learn.schedule = Learn.synchronous (index + 1) policy}, successor, call + 1, Waiting current tasks offset executed reported published finished diagnostic : accepted, next, rest)

phase :: Text -> Frame -> Bool
phase expected (Frame _ fields) = Fields.lookup "phase" fields == Just (String expected)

cycleSummary :: (Natural, Natural) -> Object -> Parser ()
cycleSummary (index, count) fields = do
    Json.fields ["phase", "index", "sessions", "seconds"] fields
    actualIndex <- fields .: "index"
    actualCount <- fields .: "sessions"
    seconds <- fields .: "seconds" >>= Json.finite
    unless (actualIndex == index && actualCount == count && seconds >= 0) (fail "Training cycle index, sessions or timing differs from its declaration")

publish :: (Run, Natural, Natural) -> (Workload.Cycle, Report.Report) -> Object -> Parser ()
publish (run, index, offset) (workload, reported) fields = do
    Json.fields ["phase", "checkpoint", "policy", "learner", "publication", "binding", "delivery"] fields
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
generations (Checked _ values _ _) = values

finalPolicy :: Checked -> Policy.Description
finalPolicy (Checked _ _ _ selected) = selected

cohort :: Generation -> Cohort.Checked
cohort (Generation value _ _ _ _ _) = value

publication :: Generation -> Value
publication (Generation _ value _ _ _ _) = Object value

diagnostics :: Generation -> [Value]
diagnostics (Generation _ _ _ values _ _) = [Object fields | Frame _ fields <- values]

stepOutputs :: Generation -> [ByteString]
stepOutputs (Generation _ _ _ values _ _) = [encoded | Frame encoded fields <- values, Fields.lookup "stage" fields `elem` map (Just . String) Step.reports]

profiles :: Generation -> [Profile.Observation]
profiles (Generation _ _ _ _ _ values) = values

describe :: Checked -> Value
describe (Checked digest accepted closed _) = object ["log_sha256" .= digest, "generations" .= map generation accepted, "closed" .= [Object fields | Frame _ fields <- closed]]
  where
    generation (Generation observed published finished diagnostic groups _) = object ["cohort" .= Cohort.describe observed, "publication" .= Object published, "cycle" .= Object finished, "diagnostics" .= [Object fields | Frame _ fields <- diagnostic], "resident_groups" .= map Group.describe groups]

closing :: Checked -> [Value]
closing (Checked _ _ values _) = [Object fields | Frame _ fields <- values]

modelLoads :: Generation -> Natural
modelLoads (Generation _ _ _ records _ _) = Execution.modelLoads records
