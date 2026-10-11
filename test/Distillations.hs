{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Distillations (distillations) where

import Control.Monad (void)
import Data.Aeson (Key, Value (..), decodeStrict, encode, object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Bifunctor (first)
import Data.ByteString.Lazy qualified as Lazy
import Data.Map.Strict qualified as Map
import Hedgehog
import Invar.Cohort qualified as C
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Learn qualified as L
import Invar.Learn.Algorithm qualified as Algorithm
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Stream qualified as S
import Invar.Learn.Worker qualified as W
import Invar.Learn.Worker.Resident qualified as Resident
import Invar.Loop qualified as Loop
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as R
import Invar.Runtime qualified as Runtime
import Invar.Spec.Invocation qualified as V
import Invar.Store qualified as Store
import Invar.Transcript qualified as Transcript
import Invar.Worker qualified as Worker
import Invar.Workload qualified as Workload
import LearnerFixture qualified as F
import Sessions qualified
import Store (workspace)
import System.Directory (canonicalizePath, createDirectory, doesDirectoryExist, doesFileExist)
import System.FilePath ((</>))

distillations :: Group
distillations =
    Group
        "Distillation learning input"
        [ ("an ungraded batch with one teacher reading per sample becomes a distillation request carrying those readings", once lowered)
        , ("a distillation input needs distillation settings and exactly one reading per sample, and a GRPO input refuses distillation settings", once refused)
        , ("the GRPO loop and runtime refuse distillation settings before they create a directory or launch a worker", once guarded)
        , ("a distillation plan runs through the checked learner exchange", once executed)
        , ("a distillation run is declared with distillation settings and tasks sampled at the teacher's temperature, and its journal declares a distillation run", once declared)
        ]
  where
    once = withTests 1 . property

teacher :: Algorithm.Teacher
teacher = Algorithm.Teacher "teacher-model" "teacher-revision" (replicate 64 '6') (L.tokenizer F.configured) (replicate 64 '7') (replicate 64 '8') (replicate 64 '9') 0.8 "none" "processed" "forced"

taught :: L.Settings
taught = F.configured {L.algorithm = L.Opd teacher}

ungraded :: R.Options -> R.OptionsOf C.Ungraded
ungraded options = options {R.definition = (R.definition options) {C.tasks = [task {C.name = show (length tasks - index), C.grade = C.Ungraded} | (index, task) <- zip [0 ..] tasks]}}
  where
    tasks = C.tasks (R.definition options)

readings :: R.BatchOf scope grade -> Map.Map String L.Reading
readings batch = Map.fromList [(R.name sample, L.Reading (V.ordinal (40 + index)) (map (const (0xbf000000 + fromIntegral index)) (Trajectory.behaviorBits (R.trajectory sample)))) | (index, sample) <- zip [0 ..] (R.samples batch)]

bindingValue :: V.Binding -> Value
bindingValue (V.Binding (V.CallId call) (V.AttemptId attempt) (V.Instance instanceId)) = object ["call" .= call, "attempt" .= attempt, "instance" .= instanceId]

lowered :: PropertyT IO ()
lowered = do
    root <- workspace
    single <- Sessions.options root 1
    outcome <- evalIO (R.withDriver (\driver -> fmap (\batch -> ([readings batch Map.! R.name sample | sample <- R.samples batch], Map.keys (readings batch) /= map R.name (R.samples batch), L.input <$> L.distilled taught (readings batch) batch)) <$> R.run driver (ungraded single)))
    (expected, permuted, encoded) <- evalEither outcome
    assert permuted
    payload <- evalEither encoded >>= evalMaybe . decodeStrict
    case payload of
        Object fields -> do
            Fields.lookup "specification" fields === Just (String "opd-token-mean/v1")
            Fields.member "delta" fields === False
            Fields.lookup "teacher" fields === Just (object (("temperature" .= (0.8 :: Double)) : [name .= value | (name, value) <- [("model", "teacher-model"), ("revision", "teacher-revision"), ("adapter", replicate 64 '6'), ("tokenizer", L.tokenizer F.configured), ("base", replicate 64 '7'), ("assembly", replicate 64 '8'), ("profile", replicate 64 '9'), ("truncation", "none"), ("logprobs", "processed"), ("path", "forced") :: (Key, String)]]))
            samples <- evalMaybe (Fields.lookup "samples" fields)
            case samples of
                Array entries -> do
                    [(Fields.lookup "teacher_bits" entry, Fields.lookup "teacher_call" entry) | Object entry <- foldr (:) [] entries] === [(Just (toJSON (L.readingWords reading)), Just (bindingValue (L.readingCall reading))) | reading <- expected]
                    assert (all (\case Object values -> not (any (`Fields.member` values) ["group", "reward", "advantage_bits"]); _ -> False) entries)
                _ -> failure
        _ -> failure

refused :: PropertyT IO ()
refused = do
    root <- workspace
    single <- Sessions.options root 1
    distilledOutcomes <- evalIO (R.withDriver (\driver -> fmap (\batch -> map (\(settings, chosen) -> void (L.distilled settings chosen batch)) (cases batch)) <$> R.run driver (ungraded single))) >>= evalEither
    distilledOutcomes === [Left L.UnmatchedReadings, Left L.UnmatchedReadings, Left (L.InvalidSettings "A distillation update needs distillation settings")]
    gradedOutcome <- evalIO (R.withDriver (\driver -> fmap (void . L.prepare taught) <$> R.run driver single)) >>= evalEither
    gradedOutcome === Left (L.InvalidSettings "A GRPO update needs GRPO settings")
  where
    cases batch =
        let chosen = readings batch
         in [(taught, Map.deleteMin chosen), (taught, Map.insert "other" (L.Reading (V.ordinal 99) [0xbf000000]) chosen), (F.configured, chosen)]

guarded :: PropertyT IO ()
guarded = do
    root <- workspace
    single <- Sessions.options root 1
    document <- evalEither Sessions.workload
    initial <- evalEither (Policy.describe ("test-model", "test-revision") (L.policy taught, L.tokenizer taught, L.behaviorBase taught, L.behaviorAssembly taught))
    evalIO (createDirectory (root </> "input") >> Policy.stageDescription (root </> "input" </> "policy.json") initial)
    let worker = R.worker single
        backend = Loop.Backend "python" (Worker.executable worker) (Worker.script worker) Nothing R.Serial "learning" W.Process root (R.sessions single)
        configured output = Loop.Config backend (root </> output) (root </> "input") (root </> "reference") taught Store.RenameExclusive
        cycle' = Loop.Cycle (C.tasks (R.definition single)) (R.order single) (R.delivery single)
    looped <- evalIO (Loop.withDriver (configured "loop") (\driver -> void (Loop.run driver cycle')))
    looped === Left (Loop.Settings (L.InvalidSettings "A GRPO update needs GRPO settings"))
    ran <- evalIO (Runtime.run (Runtime.Run (configured "runtime") 0 document) Null)
    case ran of
        Left (Runtime.Declaration problem) -> problem === "The run's workload and its learning settings name different algorithms"
        _ -> failure
    created <- evalIO (traverse (doesDirectoryExist . (root </>)) ["loop", "runtime"])
    launched <- evalIO (doesFileExist (root </> "launched"))
    (created, launched) === ([False, False], False)
executed :: PropertyT IO ()
executed = do
    root <- workspace
    single <- Sessions.options root 1
    outcome <- evalIO $ R.withDriver $ \driver -> do
        generated <- R.run driver (ungraded single)
        case first show generated >>= \batch -> first show (L.distilled taught (readings batch) batch) of
            Left problem -> pure (Left problem)
            Right planned -> do
                exchange <- F.prepare root (0, V.ordinal 7) planned
                let worker = W.Worker "/bin/sh" (root </> "process.sh") root "process checkpoint" "process reference" (Resident.output (F.paths exchange))
                writeFile (W.script worker) (F.process (root </> "replies") exchange)
                returned <- W.run worker (Transcript.echoing (const (pure ()))) (F.call exchange)
                pure (first show (fmap (length . S.completions . P.stream . W.report) returned))
    closed <- evalEither outcome
    fromIntegral closed === L.steps taught

declared :: PropertyT IO ()
declared = do
    root <- workspace
    graded <- evalEither Sessions.workload
    unanswered <- evalEither (Workload.decodeOf @C.Ungraded (Lazy.toStrict (encode [object ["tasks" .= [object ["name" .= ("a" :: String), "prompt" .= ("p" :: String), "tokens" .= (2 :: Int), "seed" .= (1 :: Int), "temperature" .= (0.8 :: Double)]], "order" .= [0 :: Int], "delivery" .= [0 :: Int]]])))
    let backend = Loop.Backend "python" "python" "inference" Nothing R.Serial "learning" W.Process root [[]]
        configured chosen = Loop.Config backend (root </> "run") (root </> "input") (root </> "reference") chosen Store.RenameExclusive
        scorer = Worker.Worker "python" "teacher" root (root </> "teacher.safetensors") [] Nothing
        cooler = taught {L.algorithm = L.Opd teacher {Algorithm.teacherTemperature = 0.7}}
    refusal "The run's workload and its learning settings name different algorithms" =<< evalIO (Runtime.run (Runtime.Run (configured taught) 0 graded) Null)
    refusal "A distillation run needs distillation settings" =<< evalIO (Runtime.distill scorer (Runtime.Run (configured F.configured) 0 unanswered) Null)
    refusal "Every task must be sampled at the teacher's temperature" =<< evalIO (Runtime.distill scorer (Runtime.Run (configured cooler) 0 unanswered) Null)
    target <- evalIO (createDirectory (root </> "run") >> canonicalizePath (root </> "run"))
    interpreted <- evalIO (Runtime.interpret target (const (Right (configured taught, 0))) (Fields.fromList ["arguments" .= Null, "workload" .= Workload.value unanswered]))
    case interpreted of
        Right (Runtime.Distilled chosen (Runtime.Run _ _ document)) -> (chosen, Workload.value document) === (teacher, Workload.value unanswered)
        _ -> failure
  where
    refusal expected = \case
        Left (Runtime.Declaration problem) -> problem === expected
        other -> annotateShow (void other) >> failure
