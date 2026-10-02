{-# LANGUAGE OverloadedStrings #-}

module Training (run, usage, settings, settingsOptions) where

import Control.Monad (when)
import Data.Aeson (Object, Value, eitherDecodeStrict, encode, object, (.:), (.=))
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Data.Map.Strict qualified as Map
import Data.Maybe (isJust, isNothing)
import Dataset qualified
import GHC.Clock (getMonotonicTime)
import InferenceInput qualified
import Invar.Learn qualified as Learn
import Invar.Learn.Protocol qualified as Protocol
import Invar.Learn.Worker qualified as Worker
import Invar.Loop qualified as Loop
import Invar.Rollout qualified as Rollout
import Invar.Runtime qualified as Runtime
import Invar.Spec.Invocation qualified as V
import Invar.Store qualified as Store
import Invar.Workload qualified as Workload
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)

run :: [String] -> IO ()
run ["--help"] = putStrLn usage
run supplied = do
    fields <- either die pure (O.parse options supplied)
    case (O.optional fields "resume", O.optional fields "staleness") of
        (Just directory, _)
            | Map.size fields == 1 -> Runtime.resume directory declared >>= either (die . show) pure
            | otherwise -> die "--resume takes no other option; the run declaration supplies them"
        (Nothing, Just _) -> do
            encoded <- Bytes.getContents
            selected <- either die pure (concurrent fields encoded)
            workload <- either die pure (eitherDecodeStrict encoded)
            Runtime.run selected ["arguments" .= supplied, "workload" .= (workload :: Value)] >>= either (die . show) pure
        (Nothing, Nothing) -> do
            config <- either die pure (configure fields)
            either (die . show) pure (Learn.validate (Loop.settings config))
            let selected = selection (Loop.settings config)
            encoded <- Bytes.getContents
            cycles <- either die pure (Dataset.decode selected encoded)
            synchronous config selected cycles

declared :: Object -> Either String Runtime.Run
declared declaration = do
    (supplied, workload) <- parseEither (\fields -> (,) <$> fields .: "arguments" <*> fields .: "workload") declaration
    fields <- O.parse options supplied
    when (isJust (O.optional fields "resume") || isNothing (O.optional fields "staleness")) (Left "The run declaration does not declare a concurrent run")
    concurrent fields (Lazy.toStrict (encode (workload :: Value)))

concurrent :: O.Fields -> Bytes.ByteString -> Either String Runtime.Run
concurrent fields encoded = do
    config <- configure fields
    first show (Learn.validate (Loop.settings config))
    lag <- O.numeric fields "staleness"
    let selected = selection (Loop.settings config)
        instantiate workloadCycle policy = Dataset.instantiate selected {Dataset.policy = policy} workloadCycle
    cycles <- Dataset.decode selected encoded
    pure (Runtime.Run config lag (map instantiate cycles) (map (fromIntegral . length . Workload.tasks) cycles))

synchronous :: Loop.Config -> Dataset.Identity -> [Dataset.Cycle] -> IO ()
synchronous config selected cycles = do
    outcome <- Loop.withDriver config $ \driver -> advance driver (length (Loop.sessions (Loop.backend config))) selected cycles
    either (die . show) pure outcome

advance :: Loop.Driver scope -> Int -> Dataset.Identity -> [Dataset.Cycle] -> IO ()
advance driver sessionCount = cycles 0
  where
    cycles :: Int -> Dataset.Identity -> [Dataset.Cycle] -> IO ()
    cycles _ _ [] = pure ()
    cycles index current (workload : rest) = do
        planned <- either die pure (Dataset.instantiate current workload)
        started <- getMonotonicTime
        generated <- Loop.run driver planned >>= either (die . show) pure
        finished <- getMonotonicTime
        let selected = Loop.current generated
        report generated
        Lazy.putStrLn (encode (object ["phase" .= ("cycle" :: String), "index" .= index, "seconds" .= (finished - started), "sessions" .= sessionCount]))
        cycles (index + 1) current {Dataset.policy = Loop.policy selected} rest

selection :: Learn.Settings -> Dataset.Identity
selection chosen = Dataset.Identity {Dataset.policy = Learn.policy chosen, Dataset.tokenizer = Learn.tokenizer chosen, Dataset.base = Learn.behaviorBase chosen, Dataset.assembly = Learn.behaviorAssembly chosen}

report :: Loop.Generation scope -> IO ()
report generated = Lazy.putStrLn (encode (object ["phase" .= ("published" :: String), "checkpoint" .= Loop.directory selected, "policy" .= Loop.policy selected, "learner" .= Loop.learner selected, "publication" .= Store.methodName (Store.method (Loop.receipt generated)), "binding" .= binding (V.completedBinding completed), "delivery" .= map binding arrivals]))
  where
    selected = Loop.current generated
    completed = Protocol.completion (Loop.result generated)
    arrivals = Rollout.delivered (Learn.rollout (Loop.plan generated))
    binding (V.Binding (V.CallId call) (V.AttemptId attempt) (V.Instance instanceId)) = object ["call" .= call, "attempt" .= attempt, "instance" .= instanceId]

configure :: O.Fields -> Either String Loop.Config
configure fields = do
    backend <- Loop.Backend <$> string "python" <*> string "inference-python" <*> string "inference" <*> pure (O.optional fields "inference-config") <*> InferenceInput.mode (O.optional fields "inference-mode") <*> string "learning" <*> learningMode (O.optional fields "learning-mode") <*> string "cache" <*> Dataset.sessions (O.optional fields "devices")
    Loop.Config backend <$> string "output" <*> string "checkpoint" <*> string "reference" <*> settings fields <*> publication fields
  where
    string = O.required fields

learningMode :: Maybe String -> Either String Worker.Mode
learningMode Nothing = Right Worker.Process
learningMode (Just "process") = Right Worker.Process
learningMode (Just "resident") = Right Worker.Resident
learningMode (Just "shared") = Right Worker.Shared
learningMode _ = Left "Invalid learning mode: expected process, resident or shared"

publication :: O.Fields -> Either String Store.Method
publication fields = do
    selected <- O.required fields "publication"
    case selected of
        "rename" -> Right Store.RenameExclusive
        "reference" -> Right Store.LinkImmutable
        _ -> Left "Invalid publication method: expected rename or reference"

settings :: O.Fields -> Either String Learn.Settings
settings fields = do
    optimizer <- Learn.Optimizer <$> number "rate" <*> number "beta1" <*> number "beta2" <*> number "optimizer-epsilon" <*> number "decay"
    current <- string "policy"
    Learn.Settings current <$> string "learner" <*> string "reference-digest" <*> string "tokenizer-digest" <*> string "base-digest" <*> string "assembly-digest" <*> string "behavior-base-digest" <*> string "behavior-assembly-digest" <*> number "clip" <*> number "penalty" <*> number "delta" <*> maybe (pure 1) (const (O.numeric fields "steps")) (O.optional fields "steps") <*> pure optimizer <*> pure (Learn.synchronous 0 current) <*> referenceSource fields
  where
    string = O.required fields
    number = O.numeric fields

referenceSource :: O.Fields -> Either String Learn.ReferenceSource
referenceSource fields = case O.optional fields "reference-source" of
    Nothing -> Right Learn.FromEngine
    Just "engine" -> Right Learn.FromEngine
    Just "learner" -> Right Learn.FromLearner
    Just _ -> Left "Invalid reference source: expected engine or learner"

usage :: String
usage = usageInfo "Usage: invar train OPTIONS < tasks.json\nAll options except --devices, --inference-config, --inference-mode, --learning-mode, --steps, --staleness and --reference-source are required. Input is a nonempty JSON array of declared cycles.\nUsage: invar train --resume DIRECTORY\nResume a run declared with --staleness from the journal in its output directory, which supplies every other option and the workload." options

options :: [OptDescr (String, String)]
options = O.descriptions [("publication", "Checkpoint publication: rename or reference"), ("devices", "Optional comma-separated CUDA devices, one rollout worker process per device"), ("python", "Learning Python executable"), ("inference-python", "Inference Python executable"), ("inference", "Inference worker script"), ("inference-config", "Optional inference worker launch configuration"), ("inference-mode", "Inference execution: serial (default), batch, resident or shared"), ("learning", "Update worker script"), ("learning-mode", "Learning execution: process (default), resident or shared; shared requires both roles"), ("cache", "Pinned model cache"), ("output", "New output directory"), ("checkpoint", "Initial paired checkpoint directory"), ("reference", "Fixed reference adapter file"), ("staleness", "Optional staleness d: run rollout and learning concurrently, update u learning from rollouts of version max(0, u - d); requires separate inference and learning processes"), ("resume", "Output directory of an interrupted run declared with --staleness; it takes no other option")] ++ settingsOptions

settingsOptions :: [OptDescr (String, String)]
settingsOptions = O.descriptions [("policy", "Consumed canonical policy tensor SHA-256"), ("tokenizer-digest", "Tokenizer operation SHA-256"), ("base-digest", "Learner frozen model tensor SHA-256"), ("assembly-digest", "Learner model assembly SHA-256"), ("behavior-base-digest", "Actual rollout frozen model SHA-256"), ("behavior-assembly-digest", "Actual rollout model assembly SHA-256"), ("learner", "Consumed learner file SHA-256"), ("reference-digest", "Canonical reference tensor SHA-256"), ("clip", "GRPO clipping coefficient"), ("penalty", "Reference penalty coefficient"), ("delta", "Advantage normalization epsilon"), ("steps", "Optional number of optimizer steps per update over consecutive mini-batches of the logical order (default 1)"), ("rate", "AdamW learning rate"), ("beta1", "AdamW first moment coefficient"), ("beta2", "AdamW second moment coefficient"), ("optimizer-epsilon", "AdamW epsilon"), ("decay", "AdamW weight decay"), ("reference-source", "Source of the objective's reference words: engine (default) or learner")]
