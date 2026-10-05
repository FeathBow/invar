{-# LANGUAGE OverloadedStrings #-}

module Invar.History (Declaration (..), Profiles (..), Checked, admit, compare, describe, rollouts) where

import Control.Monad (unless)
import Data.Aeson (Value (..), object, withObject, (.:), (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text (Text)
import Invar.History.Artifacts qualified as Artifacts
import Invar.History.Cohort qualified as Cohort
import Invar.History.Initial qualified as Initial
import Invar.History.Profile (Profiles (..))
import Invar.History.Profile qualified as Profile
import Invar.History.Publication qualified as Publication
import Invar.History.Trace qualified as Trace
import Invar.Infer qualified as Infer
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Observation qualified as Inference
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Session qualified as Session
import Invar.Json qualified as Json
import Invar.Learn qualified as Learn
import Invar.Learn.Adapter qualified as Adapter
import Invar.Learn.Codec (Decoder)
import Invar.Learn.Gradient qualified as Gradient
import Invar.Learn.Report qualified as Report
import Invar.Learn.State qualified as State
import Invar.Policy.File qualified as File
import Invar.Resident qualified as Resident
import Invar.Spec.Invocation qualified as V
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)
import Prelude hiding (compare)

data Declaration = Declaration
    { training :: Trace.Run
    , tasks :: Workload.Document
    , checkpoint :: FilePath
    , reference :: FilePath
    , randomProfile :: Initial.Random
    , initialSource :: Initial.Source
    , profileMode :: Profiles
    , finalRequest :: Infer.Request
    , finalBinding :: V.Binding
    , finalExit :: Int
    }

data Checked = Checked
    { declaration :: Declaration
    , trainingTrace :: Trace.Checked
    , initialObservation :: Value
    , generationObservations :: [Generation]
    , independentObservation :: Inference.Report
    , standaloneDiagnostics :: [Value]
    , initializationDiagnostics :: [Value]
    , profileObservation :: Value
    , initialState :: Initial.Checked
    }

data Generation = Generation {generationTrace :: Trace.Generation, generationArtifacts :: Value, generationState :: State.Observed, generationGradients :: Gradient.Observed}

admit :: Decoder -> Declaration -> (ByteString, ByteString) -> IO Checked
admit decoder declared (trainingOutput, finalOutput) = do
    trace <- either invalid pure (Trace.admit (training declared) (tasks declared) trainingOutput)
    let selected = Trace.generations trace
        settings = Trace.settings (training declared)
    final <- independent declared selected finalOutput
    admittedInitial <- Initial.admit decoder (settings, checkpoint declared, randomProfile declared) (initialSource declared)
    let schema = Initial.schema admittedInitial
        initial = Initial.describe admittedInitial
        initialDiagnostics = Initial.diagnostics admittedInitial
    File.withFile (reference declared) $ \file -> Adapter.verify (Learn.reference settings) file >> Adapter.matches schema file
    expectedRng <- either invalid pure (rng initial)
    observed <- traverse (generation (schema, expectedRng)) (zip [1 ..] selected)
    finalDiagnostics <- metadata (selected, initialDiagnostics) final finalOutput
    profiles <- either invalid pure (profileSummary declared selected (initialDiagnostics, finalDiagnostics))
    pure (Checked declared trace initial observed final finalDiagnostics initialDiagnostics profiles admittedInitial)
  where
    generation (schema, expectedRng) (index, selected) = do
        published <- Publication.observe selected
        (observed, state, gradients) <- Artifacts.successor (decoder, schema) (index, selected) published
        actualRng <- either invalid pure (rng observed)
        unless (actualRng == expectedRng) (invalid "Successor RNG vector inventory differs from the complete initial state")
        pure (Generation selected observed state gradients)

independent :: Declaration -> [Trace.Generation] -> ByteString -> IO Inference.Report
independent declared generations output = do
    lastGeneration <- case reverse generations of
        selected : _ -> pure selected
        [] -> invalid "A complete history requires a training generation"
    expected <- either invalid pure (Report.artifact "adapter" (Cohort.update (Trace.cohort lastGeneration)))
    let requested = finalRequest declared
        bound = finalBinding declared
        next = sum [fromIntegral (length (Workload.tasks workload)) + 1 | workload <- Workload.cycles (tasks declared)]
    unless (Infer.artifact requested == expected) (invalid "Final independent inference does not load the last published policy")
    either (invalid . show) pure (Learn.materialization (Trace.settings (training declared)) requested)
    unless (V.boundCall bound >= V.CallId next && V.boundAttempt bound >= V.AttemptId next && V.boundInstance bound >= V.Instance next) (invalid "Final independent inference reuses a training invocation identity")
    planned <- either (invalid . show) pure (Infer.prepare requested)
    call <- either (invalid . show) pure (Call.prepare bound planned)
    admitted <- either (invalid . show) pure (Replay.standalone Session.Single (Session.Declaration [call] Nothing) (Replay.declared (finalExit declared)) output)
    case admitted of
        [single] -> pure (Inference.view single)
        _ -> invalid "Expected one admitted final independent inference"

rng :: Value -> Either String Value
rng = parseEither (withObject "checkpoint observation" (\fields -> fields .: "state" >>= \state -> pure (Object (Fields.delete "steps" state))))

metadata :: ([Trace.Generation], [Value]) -> Inference.Report -> ByteString -> IO [Value]
metadata (generations, initial) final output = do
    expected <- either invalid pure (workerModel (Inference.describe final))
    let inferences = [Inference.describe observation | generation <- generations, observation <- Cohort.inferences (Trace.cohort generation)]
        learners = [value | generation <- generations, value@(Object fields) <- Trace.diagnostics generation, Fields.lookup "stage" fields == Just (String "loaded_learner")]
    profiles <- either invalid pure (traverse Json.decode (Bytes.lines output))
    actual <- either invalid pure (traverse workerModel (inferences ++ learners ++ selectedProfiles (initial ++ concatMap Trace.diagnostics generations ++ profiles)))
    unless (all (== expected) actual) (invalid "History workers disagree about their model or revision")
    pure [value | value@(Object fields) <- profiles, Fields.lookup "stage" fields `elem` map (Just . String) ["loading", "profile", "load", "inference"]]

selectedProfiles :: [Value] -> [Value]
selectedProfiles records = [value | value@(Object fields) <- records, Fields.lookup "stage" fields == Just (String "profile")]

profileSummary :: Declaration -> [Trace.Generation] -> ([Value], [Value]) -> Either String Value
profileSummary declared generations (initial, final) = do
    let initialProfiles = Profile.fromPrefix Resident.Learning [fields | Object fields <- initial]
        trainingProfiles = concatMap Trace.profiles generations
        mode = profileMode declared
    ending <- Profile.final mode final
    Profile.summarize mode (initialProfiles ++ trainingProfiles ++ ending)

workerModel :: Value -> Either String (Text, Text)
workerModel = parseEither (withObject "worker model observation" (\fields -> (,) <$> fields .: "model" <*> fields .: "revision"))

compare :: (Checked, Checked) -> IO Value
compare (left, right) = do
    let first = declaration left
        second = declaration right
        leftSettings = Trace.settings (training first)
        rightSettings = Trace.settings (training second)
    initial <- State.compareInitial (Initial.state (initialState left), Initial.state (initialState right))
    generations <- compareGenerations 1 (generationObservations left, generationObservations right)
    initialEqual <- decision initial
    generationEqual <- and <$> traverse decision generations
    leftModel <- either invalid pure (workerModel (Inference.describe (independentObservation left)))
    rightModel <- either invalid pure (workerModel (Inference.describe (independentObservation right)))
    let taskEqual = map Workload.tasks (Workload.cycles (tasks first)) == map Workload.tasks (Workload.cycles (tasks second))
        settingsEqual = leftSettings == rightSettings && randomProfile first == randomProfile second && initialization (initialSource first) == initialization (initialSource second)
        finalEqual = Inference.result (independentObservation left) == Inference.result (independentObservation right)
        modelEqual = leftModel == rightModel
        same = and [initialEqual, generationEqual, taskEqual, settingsEqual, finalEqual, modelEqual]
        scheduling declared = ((Trace.sessions (training declared), Trace.inferenceMode (training declared), Trace.learningMode (training declared)), map (\workload -> (Workload.order workload, Workload.delivery workload)) (Workload.cycles (tasks declared)), finalBinding declared)
        diagnostic checked = (initializationDiagnostics checked, map Trace.diagnostics (Trace.generations (trainingTrace checked)), Trace.closing (trainingTrace checked), standaloneDiagnostics checked)
    pure (object ["comparison" .= ("complete histories: semantic inputs and artifact values" :: Text), "equal" .= same, "tasks_equal" .= taskEqual, "settings_equal" .= settingsEqual, "models_equal" .= modelEqual, "schedule_equal" .= (scheduling first == scheduling second), "publication_method_equal" .= (Trace.method (training first) == Trace.method (training second)), "diagnostics_equal" .= (diagnostic left == diagnostic right), "initial" .= initial, "generations" .= generations, "final_equal" .= finalEqual, "left" .= describe left, "right" .= describe right])

initialization :: Initial.Source -> Maybe Integer
initialization Initial.Provided = Nothing
initialization (Initial.Executed run _) = Just (Initial.seed run)

compareGenerations :: Natural -> ([Generation], [Generation]) -> IO [Value]
compareGenerations _ ([], []) = pure []
compareGenerations index ([], remaining) = pure (missing "left" index remaining)
compareGenerations index (remaining, []) = pure (missing "right" index remaining)
compareGenerations index (left : restLeft, right : restRight) = do
    observed <- compareGeneration index (left, right)
    rest <- compareGenerations (index + 1) (restLeft, restRight)
    pure (observed : rest)

missing :: Text -> Natural -> [Generation] -> [Value]
missing side index remaining = [object ["generation" .= position, "missing" .= side, "equal" .= False] | (position, _) <- zip [index ..] remaining]

compareGeneration :: Natural -> (Generation, Generation) -> IO Value
compareGeneration index (left, right) = do
    let leftCohort = Trace.cohort (generationTrace left)
        rightCohort = Trace.cohort (generationTrace right)
    inputsEqual <- either invalid pure (Report.sameInput (Cohort.update leftCohort) (Cohort.update rightCohort))
    states <- State.compareObserved (generationState left, generationState right)
    gradients <- Gradient.compareObserved (generationGradients left, generationGradients right)
    statesEqual <- decision states
    gradientsEqual <- decision gradients
    leftProbabilities <- field "probabilities" (generationArtifacts left)
    rightProbabilities <- field "probabilities" (generationArtifacts right)
    let probabilitiesEqual = leftProbabilities == rightProbabilities
        inferenceEqual = map Inference.result (Cohort.inferences leftCohort) == map Inference.result (Cohort.inferences rightCohort)
    pure (object ["generation" .= index, "equal" .= and [inputsEqual, statesEqual, gradientsEqual, probabilitiesEqual, inferenceEqual], "inputs_equal" .= inputsEqual, "inferences_equal" .= inferenceEqual, "states" .= states, "gradients" .= gradients, "probabilities_equal" .= probabilitiesEqual])

decision :: Value -> IO Bool
decision = either invalid pure . parseEither (withObject "comparison decision" (.: "equal"))

field :: Key -> Value -> IO Value
field key = either invalid pure . parseEither (withObject "artifact observation" (.: key))

rollouts :: Checked -> Natural -> Either String [(String, Inference.Report)]
rollouts checked index = case drop (fromIntegral index - 1) (zip (Workload.cycles (tasks (declaration checked))) (generationObservations checked)) of
    (workload, generation) : _ | index >= 1 -> Right (zip (map Workload.name (Workload.tasks workload)) (Cohort.inferences (Trace.cohort (generationTrace generation))))
    _ -> Left "The admitted history has no such generation"

describe :: Checked -> Value
describe checked = object ["tasks_sha256" .= Workload.digest (tasks (declaration checked)), "initial" .= initialObservation checked, "initial_diagnostics" .= initializationDiagnostics checked, "training" .= Trace.describe (trainingTrace checked), "artifacts" .= map generationArtifacts (generationObservations checked), "final" .= Inference.describe (independentObservation checked), "final_diagnostics" .= standaloneDiagnostics checked, "profiles" .= profileObservation checked]

invalid :: String -> IO value
invalid = ioError . userError
