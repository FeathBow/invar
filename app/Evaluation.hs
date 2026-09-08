{-# LANGUAGE OverloadedStrings #-}

module Evaluation (run) where

import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value, encode, object, (.=))
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Data.Map.Strict qualified as Map
import Dataset qualified
import Invar.Artifact qualified as Artifact
import Invar.Cohort qualified as Cohort
import Invar.Infer qualified as Infer
import Invar.Infer.Result qualified as Result
import Invar.Loop qualified as Loop
import Invar.Rollout qualified as Rollout
import Invar.Spec.Invocation qualified as Invocation
import Invar.Worker qualified as Worker
import Numeric.Natural (Natural)
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)

run :: [String] -> IO ()
run ["--help"] = putStrLn usage
run supplied = do
    (worker, selected, sessions) <- either die pure (O.parse options supplied >>= configure)
    encoded <- Bytes.getContents
    workloads <- either die pure (Dataset.decode selected encoded)
    plans <- either die pure (traverse (prepare (worker, sessions) selected) workloads)
    Rollout.withDriver $ \driver -> mapM_ (evaluate driver) (zip [0 :: Natural ..] plans)
    emit (object ["phase" .= ("evaluation_complete" :: String), "policy" .= Dataset.policy selected, "tokenizer" .= Dataset.tokenizer selected, "base" .= Dataset.base selected, "assembly" .= Dataset.assembly selected, "cohorts" .= length plans, "sessions" .= length sessions, "tasks_sha256" .= Artifact.hex (SHA256.hash encoded)])

prepare :: (Worker.Worker, [[(String, String)]]) -> Dataset.Identity -> Dataset.Cycle -> Either String Rollout.Options
prepare (worker, sessions) selected workload = do
    planned <- Dataset.instantiate selected workload
    pure Rollout.Options {Rollout.worker = worker, Rollout.sessions = sessions, Rollout.definition = Cohort.Definition (Dataset.policy selected) (Loop.tasks planned), Rollout.order = Loop.order planned, Rollout.delivery = Loop.delivery planned}

evaluate :: Rollout.Driver scope -> (Natural, Rollout.Options) -> IO ()
evaluate driver (index, planned) = do
    completed <- Rollout.run driver planned >>= either (die . show) pure
    emit (object ["phase" .= ("evaluation" :: String), "cohort" .= index, "policy" .= Cohort.policy (Rollout.definition planned), "summary" .= summary completed, "samples" .= map sample (Rollout.samples completed)])

summary :: Rollout.Batch scope -> Value
summary batch = object ["sample_count" .= length values, "reward_sum" .= (fromRational (sum rewards) :: Double), "response_tokens" .= sum (map responseLength observations), "truncated_count" .= length (filter Result.truncated observations), "group_count" .= Map.size groups, "zero_variance_groups" .= length (filter constant (Map.elems groups))]
  where
    values = Rollout.samples batch
    rewards = map Rollout.reward values
    observations = map Rollout.observation values
    groups = Map.fromListWith (++) [(Rollout.group value, [Rollout.reward value]) | value <- values]
    constant [] = False
    constant (first : rest) = all (== first) rest

sample :: Rollout.Sample -> Value
sample value = object ["name" .= Rollout.name value, "group" .= Rollout.group value, "seed" .= Infer.seed (Result.consumed observed), "reward" .= (fromRational (Rollout.reward value) :: Double), "response_tokens" .= responseLength observed, "truncated" .= Result.truncated observed, "binding" .= object ["call" .= call, "attempt" .= attempt, "instance" .= instanceId]]
  where
    observed = Rollout.observation value
    Invocation.Binding (Invocation.CallId call) (Invocation.AttemptId attempt) (Invocation.Instance instanceId) = Invocation.completedBinding (Rollout.completion value)

responseLength :: Result.Result -> Int
responseLength = length . Result.behaviorBits

emit :: Value -> IO ()
emit = Lazy.putStrLn . encode

configure :: O.Fields -> Either String (Worker.Worker, Dataset.Identity, [[(String, String)]])
configure fields = do
    worker <- Worker.Worker <$> string "python" <*> string "worker" <*> string "cache" <*> string "adapter" <*> pure []
    selected <- Dataset.Identity <$> string "policy" <*> string "tokenizer-digest" <*> string "base-digest" <*> string "assembly-digest"
    sessions <- Dataset.sessions (O.optional fields "devices")
    pure (worker, selected, sessions)
  where
    string = O.required fields

usage :: String
usage = usageInfo "Usage: invar evaluate OPTIONS < tasks.json\nAll options are required. Every cohort uses the same policy; no learning or publication occurs." options

options :: [OptDescr (String, String)]
options = O.descriptions [("devices", "Optional comma-separated CUDA devices, one worker process per device"), ("python", "Python executable"), ("worker", "Batch inference worker script (session.py)"), ("cache", "Pinned model cache"), ("adapter", "Adapter file"), ("policy", "Canonical adapter tensor SHA-256"), ("tokenizer-digest", "Tokenizer operation SHA-256"), ("base-digest", "Frozen model tensor SHA-256"), ("assembly-digest", "Model assembly SHA-256")]
