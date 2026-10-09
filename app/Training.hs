{-# LANGUAGE OverloadedStrings #-}

module Training (run, usage, settings, settingsOptions, declared) where

import Control.Monad (when)
import Data.Aeson (Value, encode, object, parseJSON, toJSON, (.=))
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Data.Map.Strict qualified as Map
import Data.Maybe (isJust, isNothing)
import Data.Text qualified as Text
import Dataset qualified
import GHC.Clock (getMonotonicTime)
import InferenceInput qualified
import Invar.Learn qualified as Learn
import Invar.Learn.Protocol qualified as Protocol
import Invar.Learn.Stream qualified as Stream
import Invar.Learn.Worker qualified as Worker
import Invar.Loop qualified as Loop
import Invar.Rollout qualified as Rollout
import Invar.Runtime qualified as Runtime
import Invar.Spec.Invocation qualified as V
import Invar.Store qualified as Store
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)
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
        (Nothing, staleness) -> do
            config <- either die pure (configure fields)
            either (die . show) pure (Learn.validate (Loop.settings config))
            if isNothing staleness && shared config
                then do
                    let selected = selection (Loop.settings config)
                    encoded <- Bytes.getContents
                    cycles <- either die pure (Dataset.decode selected encoded)
                    synchronous config selected cycles
                else do
                    chosen <- either die pure (lagOf fields)
                    document <- Bytes.getContents >>= either die pure . Workload.decode
                    Runtime.run (Runtime.Run config chosen document) (toJSON supplied) >>= either (die . show) pure

declared :: Value -> Either String (Loop.Config, Natural)
declared arguments = do
    fields <- parseEither parseJSON arguments >>= O.parse options
    when (isJust (O.optional fields "resume")) (Left "A resume is not a run declaration")
    concurrent fields

concurrent :: O.Fields -> Either String (Loop.Config, Natural)
concurrent fields = do
    config <- configure fields
    first show (Learn.validate (Loop.settings config))
    chosen <- lagOf fields
    pure (config, chosen)

-- A run without --staleness runs the event runtime at staleness zero; the lockstep driver is kept for a shared run, which uses it until #26 Part B moves shared execution over.
lagOf :: O.Fields -> Either String Natural
lagOf fields = case O.optional fields "staleness" of
    Nothing -> Right 0
    Just _ -> O.numeric fields "staleness"

shared :: Loop.Config -> Bool
shared config = Loop.inferenceMode (Loop.backend config) == Rollout.Shared || Loop.learningMode (Loop.backend config) == Worker.Shared

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
    Just named -> maybe (Left "Invalid reference source: expected engine or learner") Right (Stream.readSource (Text.pack named))

usage :: String
usage = usageInfo "Usage: invar train OPTIONS < tasks.json\nAll options except --devices, --inference-config, --inference-mode, --learning-mode, --steps, --staleness and --reference-source are required. Input is a nonempty JSON array of declared cycles. A non-shared run uses the event runtime at staleness zero unless --staleness says otherwise; a shared run uses the lockstep driver and takes an explicit --staleness 0 to use the runtime instead.\nUsage: invar train --resume DIRECTORY\nResume an interrupted run of the event runtime from the journal in its output directory, which supplies every other option and the workload. Every run of the event runtime keeps a journal; a shared run without --staleness runs on the lockstep driver and keeps none." options

options :: [OptDescr (String, String)]
options = O.descriptions [("publication", "Checkpoint publication: rename or reference"), ("devices", "Optional comma-separated CUDA devices, one rollout worker process per device"), ("python", "Learning Python executable"), ("inference-python", "Inference Python executable"), ("inference", "Inference worker script"), ("inference-config", "Optional inference worker launch configuration"), ("inference-mode", "Inference execution: serial (default), batch, resident or shared"), ("learning", "Update worker script"), ("learning-mode", "Learning execution: process (default), resident or shared; shared requires both roles"), ("cache", "Pinned model cache"), ("output", "New output directory"), ("checkpoint", "Initial paired checkpoint directory"), ("reference", "Fixed reference adapter file"), ("staleness", "Optional staleness d (default 0): run through the event runtime, update u learning from rollouts of version max(0, u - d) while later rollouts may run; shared inference and learning run only with d = 0 and only reach the runtime with this option"), ("resume", "Output directory of an interrupted run of the event runtime; it takes no other option")] ++ settingsOptions

settingsOptions :: [OptDescr (String, String)]
settingsOptions = O.descriptions [("policy", "Consumed canonical policy tensor SHA-256"), ("tokenizer-digest", "Tokenizer operation SHA-256"), ("base-digest", "Learner frozen model tensor SHA-256"), ("assembly-digest", "Learner model assembly SHA-256"), ("behavior-base-digest", "Actual rollout frozen model SHA-256"), ("behavior-assembly-digest", "Actual rollout model assembly SHA-256"), ("learner", "Consumed learner file SHA-256"), ("reference-digest", "Canonical reference tensor SHA-256"), ("clip", "GRPO clipping coefficient"), ("penalty", "Reference penalty coefficient"), ("delta", "Advantage normalization epsilon"), ("steps", "Optional number of optimizer steps per update over consecutive mini-batches of the logical order (default 1)"), ("rate", "AdamW learning rate"), ("beta1", "AdamW first moment coefficient"), ("beta2", "AdamW second moment coefficient"), ("optimizer-epsilon", "AdamW epsilon"), ("decay", "AdamW weight decay"), ("reference-source", "Source of the objective's reference words: engine (default) or learner")]
