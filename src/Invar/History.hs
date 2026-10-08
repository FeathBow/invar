{-# LANGUAGE OverloadedStrings #-}

module Invar.History (Declaration (..), Training (..), Log (..), Profiles (..), Checked, admit, compare, describe, rollouts) where

import Control.Monad (unless)
import Data.Aeson (Value (..), object, withObject, (.:), (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.List (genericLength)
import Data.Text (Text)
import Invar.History.Artifacts qualified as Artifacts
import Invar.History.Cohort qualified as Cohort
import Invar.History.Generation qualified as Generation
import Invar.History.Initial qualified as Initial
import Invar.History.Profile (Profiles (..))
import Invar.History.Profile qualified as Profile
import Invar.History.Publication qualified as Publication
import Invar.History.Runtime qualified as Runtime
import Invar.History.Schedule qualified as Schedule
import Invar.History.Trace qualified as Trace
import Invar.Infer qualified as Infer
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Observation qualified as Inference
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Session qualified as Session
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Learn qualified as Learn
import Invar.Learn.Adapter qualified as Adapter
import Invar.Learn.Codec (Decoder)
import Invar.Learn.Gradient qualified as Gradient
import Invar.Learn.Report qualified as Report
import Invar.Learn.Request qualified as Request
import Invar.Learn.State qualified as State
import Invar.Learn.Stream qualified as S
import Invar.Learn.Worker qualified as W
import Invar.Loop qualified as Loop
import Invar.Policy qualified as Policy
import Invar.Policy.File qualified as File
import Invar.Resident qualified as Resident
import Invar.Rollout qualified as R
import Invar.Spec.Invocation qualified as V
import Invar.Store qualified as Store
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)
import System.FilePath ((</>))
import Prelude hiding (compare)

data Log = Log {trainingRun :: Trace.Run, tasks :: Workload.Document, checkpoint :: FilePath, reference :: FilePath, trainingOutput :: ByteString}

data Training = Logged Log | Recorded FilePath (Value -> Either String (Loop.Config, Natural))

data Declaration = Declaration
    { training :: Training
    , randomProfile :: Initial.Random
    , initialSource :: Initial.Source
    , profileMode :: Profiles
    , finalRequest :: Infer.Request
    , finalBinding :: V.Binding
    , finalExit :: Int
    }

data Checked = Checked
    { declaration :: Declaration
    , trainingRecord :: Source
    , initialObservation :: Value
    , generationObservations :: [Generation]
    , independentObservation :: Inference.Report
    , standaloneDiagnostics :: [Value]
    , initializationDiagnostics :: [Value]
    , profileObservation :: Value
    , initialState :: Initial.Checked
    }

data Source = FromLog Log Trace.Checked | FromRun Runtime.Checked

data Arrangement = Arrangement {sessionCount :: Natural, inferenceRun :: Trace.Mode, learningRun :: Trace.Mode, dispatch :: [([Natural], [Natural])], finalCall :: V.Binding}
    deriving (Eq)

data Generation = Generation {generationTrace :: Generation.Generation, generationArtifacts :: Value, generationState :: State.Observed, generationGradients :: Gradient.Observed}

admit :: Decoder -> Declaration -> ByteString -> IO Checked
admit decoder declared finalOutput = do
    (source, selected, finalPolicy, next, (trainingProfiles, loaded)) <- case training declared of
        Logged logged -> do
            initialPolicy <- Policy.readDescription (checkpoint logged </> "policy.json")
            trace <- either invalid pure (Trace.admit (trainingRun logged) initialPolicy (tasks logged) (trainingOutput logged))
            let generations = Trace.generations trace
            pure (FromLog logged trace, generations, Trace.finalPolicy trace, sum [fromIntegral (length (Workload.tasks workload)) + 1 | workload <- Workload.cycles (tasks logged)], (concatMap Generation.profiles generations, []))
        Recorded directory interpreter -> do
            checked <- Runtime.inspect directory interpreter >>= either invalid pure
            pure (FromRun checked, Runtime.generations checked, Runtime.finalPolicy checked, Runtime.identities checked, (Runtime.profiles checked, Runtime.loads checked))
    let settings = settingsOf source
    lastGeneration <- case reverse selected of
        chosen : _ -> pure chosen
        [] -> invalid "A complete history requires a training generation"
    final <- independent declared (settings, lastGeneration, finalPolicy, next) finalOutput
    admittedInitial <- Initial.admit decoder (settings, checkpointOf source, randomProfile declared) (initialSource declared)
    let schema = Initial.schema admittedInitial
        initial = Initial.describe admittedInitial
        initialDiagnostics = Initial.diagnostics admittedInitial
    File.withFile (referenceOf source) $ \file -> Adapter.verify (Learn.reference settings) file >> Adapter.matches schema file
    expectedRng <- either invalid pure (rng initial)
    observed <- traverse (generation (schema, expectedRng)) (zip (drop 1 (scanl (+) 0 (map optimizerSteps selected))) selected)
    finalDiagnostics <- metadata (selected, initialDiagnostics ++ loaded) final finalOutput
    profiles <- either invalid pure (profileSummary declared trainingProfiles (initialDiagnostics, finalDiagnostics))
    pure (Checked declared source initial observed final finalDiagnostics initialDiagnostics profiles admittedInitial)
  where
    generation (schema, expectedRng) (expected, selected) = do
        published <- Publication.observe selected
        (observed, state, gradients) <- Artifacts.successor (decoder, schema) (expected, selected) published
        actualRng <- either invalid pure (rng observed)
        unless (actualRng == expectedRng) (invalid "Successor RNG vector inventory differs from the complete initial state")
        pure (Generation selected observed state gradients)

settingsOf :: Source -> Learn.Settings
settingsOf (FromLog logged _) = Trace.settings (trainingRun logged)
settingsOf (FromRun run) = Loop.settings (Runtime.config run)

methodOf :: Source -> String
methodOf (FromLog logged _) = Trace.method (trainingRun logged)
methodOf (FromRun run) = Store.methodName (Loop.publication (Runtime.config run))

workloadOf :: Source -> Workload.Document
workloadOf (FromLog logged _) = tasks logged
workloadOf (FromRun run) = Runtime.workload run

checkpointOf :: Source -> FilePath
checkpointOf (FromLog logged _) = checkpoint logged
checkpointOf (FromRun run) = Loop.checkpoint (Runtime.config run)

referenceOf :: Source -> FilePath
referenceOf (FromLog logged _) = reference logged
referenceOf (FromRun run) = Loop.reference (Runtime.config run)

optimizerSteps :: Generation.Generation -> Integer
optimizerSteps = genericLength . S.batches . Request.exchange . Report.checkedRequest . Cohort.update . Generation.cohort

independent :: Declaration -> (Learn.Settings, Generation.Generation, Policy.Description, Natural) -> ByteString -> IO Inference.Report
independent declared (settings, lastGeneration, finalPolicy, next) output = do
    expected <- either invalid pure (Report.artifact "adapter" (Cohort.update (Generation.cohort lastGeneration)))
    let requested = finalRequest declared
        bound = finalBinding declared
    unless (Infer.artifact requested == expected) (invalid "Final independent inference does not load the last published policy")
    either (invalid . show) pure (Learn.materialization settings requested)
    unless (V.boundCall bound >= V.CallId next && V.boundAttempt bound >= V.AttemptId next && V.boundInstance bound >= V.Instance next) (invalid "Final independent inference reuses a training invocation identity")
    planned <- either (invalid . show) pure (Infer.prepare requested >>= Infer.bindPolicy finalPolicy)
    call <- either (invalid . show) pure (Call.prepare bound planned)
    admitted <- either (invalid . show) pure (Replay.standalone Session.Single (Session.Declaration [call] Nothing) (Replay.declared (finalExit declared)) output)
    case admitted of
        [single] -> pure (Inference.view single)
        _ -> invalid "Expected one admitted final independent inference"

rng :: Value -> Either String Value
rng = parseEither (withObject "checkpoint observation" (\fields -> fields .: "state" >>= \state -> pure (Object (Fields.delete "steps" state))))

metadata :: ([Generation.Generation], [Value]) -> Inference.Report -> ByteString -> IO [Value]
metadata (generations, initial) final output = do
    expected <- either invalid pure (workerModel (Inference.describe final))
    let inferences = [Inference.describe observation | generation <- generations, observation <- Cohort.inferences (Generation.cohort generation)]
        learners = [value | generation <- generations, value@(Object fields) <- Generation.diagnostics generation, Fields.lookup "stage" fields == Just (String "loaded_learner")]
    profiles <- either invalid pure (traverse Json.decode (Bytes.lines output))
    actual <- either invalid pure (traverse workerModel (inferences ++ learners ++ selectedProfiles (initial ++ concatMap Generation.diagnostics generations ++ profiles)))
    unless (all (== expected) actual) (invalid "History workers disagree about their model or revision")
    pure [value | value@(Object fields) <- profiles, Fields.lookup "stage" fields `elem` map (Just . String) ["loading", "profile", "load", "inference"]]

selectedProfiles :: [Value] -> [Value]
selectedProfiles records = [value | value@(Object fields) <- records, Fields.lookup "stage" fields == Just (String "profile")]

profileSummary :: Declaration -> [Profile.Observation] -> ([Value], [Value]) -> Either String Value
profileSummary declared trainingProfiles (initial, final) = do
    let initialProfiles = Profile.fromPrefix Resident.Learning [fields | Object fields <- initial]
        mode = profileMode declared
    ending <- Profile.final mode final
    Profile.summarize mode (initialProfiles ++ trainingProfiles ++ ending)

workerModel :: Value -> Either String (Text, Text)
workerModel = parseEither (withObject "worker model observation" (\fields -> (,) <$> fields .: "model" <*> fields .: "revision"))

compare :: (Checked, Checked) -> IO Value
compare (left, right) = do
    let first = declaration left
        second = declaration right
        leftSettings = settingsOf (trainingRecord left)
        rightSettings = settingsOf (trainingRecord right)
    initial <- State.compareInitial (Initial.state (initialState left), Initial.state (initialState right))
    generations <- compareGenerations 1 (generationObservations left, generationObservations right)
    initialEqual <- decision initial
    generationEqual <- and <$> traverse decision generations
    leftModel <- either invalid pure (workerModel (Inference.describe (independentObservation left)))
    rightModel <- either invalid pure (workerModel (Inference.describe (independentObservation right)))
    let taskEqual = map Workload.tasks (Workload.cycles (workloadOf (trainingRecord left))) == map Workload.tasks (Workload.cycles (workloadOf (trainingRecord right)))
        settingsEqual = leftSettings == rightSettings && randomProfile first == randomProfile second && initialization (initialSource first) == initialization (initialSource second)
        finalEqual = Inference.result (independentObservation left) == Inference.result (independentObservation right)
        modelEqual = leftModel == rightModel
        leftSchedule = schedule left
        rightSchedule = schedule right
        scheduleEqual = leftSchedule == rightSchedule
        same = and [initialEqual, generationEqual, taskEqual, settingsEqual, scheduleEqual, finalEqual, modelEqual]
        executed checked = object ["declared" .= describeArrangement (arrangement checked), "recorded" .= recorded (trainingRecord checked)]
        diagnostic checked = (initializationDiagnostics checked, map Generation.diagnostics (trained checked), closing (trainingRecord checked), standaloneDiagnostics checked)
    pure (object ["comparison" .= ("complete histories: semantic inputs and artifact values" :: Text), "equal" .= same, "tasks_equal" .= taskEqual, "settings_equal" .= settingsEqual, "models_equal" .= modelEqual, "schedule_equal" .= scheduleEqual, "schedule" .= object ["left" .= Schedule.describe leftSchedule, "right" .= Schedule.describe rightSchedule, "differences" .= Schedule.differences leftSchedule rightSchedule], "execution" .= object ["declared_equal" .= (arrangement left == arrangement right), "left" .= executed left, "right" .= executed right], "publication_method_equal" .= (methodOf (trainingRecord left) == methodOf (trainingRecord right)), "diagnostics_equal" .= (diagnostic left == diagnostic right), "initial" .= initial, "generations" .= generations, "final_equal" .= finalEqual, "left" .= describe left, "right" .= describe right])

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
    let leftCohort = Generation.cohort (generationTrace left)
        rightCohort = Generation.cohort (generationTrace right)
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
rollouts checked index = case drop (fromIntegral index - 1) (zip (Workload.cycles (workloadOf (trainingRecord checked))) (generationObservations checked)) of
    (workload, generation) : _ | index >= 1 -> Right (zip (map Workload.name (Workload.tasks workload)) (Cohort.inferences (Generation.cohort (generationTrace generation))))
    _ -> Left "The admitted history has no such generation"

describe :: Checked -> Value
describe checked = object ["tasks_sha256" .= Workload.digest (workloadOf (trainingRecord checked)), "initial" .= initialObservation checked, "initial_diagnostics" .= initializationDiagnostics checked, "training" .= describeSource (trainingRecord checked), "schedule" .= Schedule.describe (schedule checked), "artifacts" .= map generationArtifacts (generationObservations checked), "final" .= Inference.describe (independentObservation checked), "final_diagnostics" .= standaloneDiagnostics checked, "profiles" .= profileObservation checked]

trained :: Checked -> [Generation.Generation]
trained checked = map generationTrace (generationObservations checked)

closing :: Source -> [Value]
closing (FromLog _ trace) = Trace.closing trace
closing (FromRun _) = []

describeSource :: Source -> Value
describeSource (FromLog _ trace) = Trace.describe trace
describeSource (FromRun run) = Runtime.describe run

schedule :: Checked -> Schedule.Schedule
schedule checked = Schedule.project [Report.checkedRequest (Cohort.update (Generation.cohort generation)) | generation <- trained checked]

arrangement :: Checked -> Arrangement
arrangement checked = case trainingRecord checked of
    FromLog logged _ -> let run = trainingRun logged in Arrangement (Trace.sessions run) (Trace.inferenceMode run) (Trace.learningMode run) cycles final
    FromRun run ->
        let backend = Loop.backend (Runtime.config run)
         in Arrangement (genericLength (Loop.sessions backend)) (rollout (Loop.inferenceMode backend)) (learner (Loop.learningMode backend)) cycles final
  where
    cycles = [(Workload.order workload, Workload.delivery workload) | workload <- Workload.cycles (workloadOf (trainingRecord checked))]
    final = finalBinding (declaration checked)
    rollout R.Serial = Trace.Finite
    rollout R.Batched = Trace.Batched
    rollout R.Resident = Trace.Resident
    rollout R.Shared = Trace.Shared
    learner W.Process = Trace.Finite
    learner W.Resident = Trace.Resident
    learner W.Shared = Trace.Shared

describeArrangement :: Arrangement -> Value
describeArrangement arranged = object ["sessions" .= sessionCount arranged, "inference" .= show (inferenceRun arranged), "learning" .= show (learningRun arranged), "cycles" .= [object ["order" .= executionOrder, "delivery" .= arrival] | (executionOrder, arrival) <- dispatch arranged], "final_binding" .= Wire.bindingValue (finalCall arranged)]

recorded :: Source -> Value
recorded (FromLog _ _) = object ["source" .= ("training log" :: Text)]
recorded (FromRun run) = Runtime.recorded run

invalid :: String -> IO value
invalid = ioError . userError
