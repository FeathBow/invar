{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

module Distillations (distillations) where

import Control.Monad (void)
import Data.Aeson (Key, Value (..), decodeStrict, object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Map.Strict qualified as Map
import Hedgehog
import Invar.Cohort qualified as C
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Learn qualified as L
import Invar.Learn.Algorithm qualified as Algorithm
import Invar.Learn.Worker qualified as W
import Invar.Loop qualified as Loop
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as R
import Invar.Runtime qualified as Runtime
import Invar.Spec.Invocation qualified as V
import Invar.Store qualified as Store
import Invar.Worker qualified as Worker
import LearnerFixture qualified as F
import Sessions qualified
import Store (workspace)
import System.Directory (createDirectory, doesDirectoryExist, doesFileExist)
import System.FilePath ((</>))

distillations :: Group
distillations =
    Group
        "Distillation learning input"
        [ ("an ungraded batch with one teacher reading per sample becomes a distillation request carrying those readings", once lowered)
        , ("a distillation input needs distillation settings and exactly one reading per sample, and a GRPO input refuses distillation settings", once refused)
        , ("the GRPO loop and runtime refuse distillation settings before they create a directory or launch a worker", once guarded)
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
