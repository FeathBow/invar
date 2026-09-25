{-# LANGUAGE OverloadedStrings #-}

module Evaluation (run) where

import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value, encode, object, (.=))
import Data.Bifunctor (first)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Dataset qualified
import InferenceInput qualified
import Invar.Artifact qualified as Artifact
import Invar.Cohort qualified as Cohort
import Invar.Evaluation qualified as Evaluation
import Invar.Infer qualified as Infer
import Invar.Loop qualified as Loop
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as Rollout
import Invar.Worker qualified as Worker
import Numeric.Natural (Natural)
import Options qualified as O
import PolicyInput qualified
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)

run :: [String] -> IO ()
run ["--help"] = putStrLn usage
run supplied = do
    fields <- either die pure (O.parse options supplied)
    selection@(_, description) <- PolicyInput.selection "policy" fields
    (worker, selected, sessions, mode) <- either die pure (configure selection fields)
    encoded <- Bytes.getContents
    workloads <- either die pure (Dataset.decode selected encoded)
    plans <- either die pure (traverse (prepare (worker, sessions, mode) (selected, description)) workloads)
    completed <- Rollout.withConfiguredDriver mode (worker, sessions) (\driver -> mapM_ (evaluate driver) (zip [0 :: Natural ..] plans))
    either (die . show) pure completed
    emit (object (["phase" .= ("evaluation_complete" :: String), "policy" .= Dataset.policy selected, "tokenizer" .= Dataset.tokenizer selected, "base" .= Dataset.base selected, "assembly" .= Dataset.assembly selected, "cohorts" .= length plans, "sessions" .= length sessions, "tasks_sha256" .= Artifact.hex (SHA256.hash encoded)] ++ ["worker_mode" .= ("resident" :: String) | mode == Rollout.Resident]))

prepare :: (Worker.Worker, [[(String, String)]], Rollout.Mode) -> (Dataset.Identity, Maybe Policy.Description) -> Dataset.Cycle -> Either String Rollout.Options
prepare (worker, sessions, mode) (selected, description) workload = do
    planned <- Dataset.instantiate selected workload
    tasks <- traverse bind (Loop.tasks planned)
    pure Rollout.Options {Rollout.worker = worker, Rollout.mode = mode, Rollout.sessions = sessions, Rollout.definition = Cohort.Definition (Dataset.policy selected) tasks, Rollout.order = Loop.order planned, Rollout.delivery = Loop.delivery planned, Rollout.reference = Nothing}
  where
    bind task = do
        requested <- first show (maybe Right Infer.bindPolicy description (Cohort.plan task))
        pure task {Cohort.plan = requested}

evaluate :: Rollout.Driver scope -> (Natural, Rollout.Options) -> IO ()
evaluate driver (index, planned) = do
    completed <- Rollout.run driver planned >>= either (die . show) pure
    either die emit (Evaluation.cohortValue index (Cohort.policy (Rollout.definition planned)) completed)

emit :: Value -> IO ()
emit = Lazy.putStrLn . encode

configure :: (FilePath, Maybe Policy.Description) -> O.Fields -> Either String (Worker.Worker, Dataset.Identity, [[(String, String)]], Rollout.Mode)
configure (adapter, description) fields = do
    worker <- Worker.Worker <$> string "python" <*> string "worker" <*> string "cache" <*> pure adapter <*> pure [] <*> pure (O.optional fields "worker-config")
    selected <- case description of
        Nothing -> Dataset.Identity <$> string "policy" <*> string "tokenizer-digest" <*> string "base-digest" <*> string "assembly-digest"
        Just policy -> pure (Dataset.Identity (Policy.adapter policy) (Policy.tokenizer policy) (Policy.base policy) (Policy.assembly policy))
    sessions <- Dataset.sessions (O.optional fields "devices")
    mode <- InferenceInput.mode (O.optional fields "worker-mode")
    pure (worker, selected, sessions, mode)
  where
    string = O.required fields

usage :: String
usage = usageInfo "Usage: invar evaluate OPTIONS < tasks.json\nSelect --checkpoint, or an explicit --adapter with --policy and all three materialization digests. --checkpoint derives the inference selection from policy.json and binds every task before dispatch. --devices, --worker-config and --worker-mode are optional. Every cohort uses the same policy; no learning or publication occurs." options

options :: [OptDescr (String, String)]
options = O.descriptions [("devices", "Optional comma-separated CUDA devices, one worker process per device"), ("python", "Python executable"), ("worker", "Inference worker script"), ("worker-config", "Optional worker launch configuration"), ("worker-mode", "Inference execution: serial (default), batch or resident"), ("cache", "Pinned model cache"), ("checkpoint", "Checkpoint containing policy.json and adapter.safetensors"), ("adapter", "Explicit adapter file or native handoff directory"), ("policy", "Canonical adapter tensor SHA-256"), ("tokenizer-digest", "Tokenizer operation SHA-256"), ("base-digest", "Frozen model tensor SHA-256"), ("assembly-digest", "Model assembly SHA-256")]
