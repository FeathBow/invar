{-# LANGUAGE OverloadedStrings #-}

module Training (run, usage, settings, settingsOptions, declared) where

import Control.Monad (when)
import Data.Aeson (Value, encode, object, parseJSON, toJSON, (.=))
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Data.Map.Strict qualified as Map
import Data.Maybe (isJust)
import Data.Text qualified as Text
import Dataset qualified
import GHC.Clock (getMonotonicTime)
import InferenceInput qualified
import Invar.Cohort qualified as Cohort
import Invar.Learn qualified as Learn
import Invar.Learn.Algorithm qualified as Algorithm
import Invar.Learn.Protocol qualified as Protocol
import Invar.Learn.Stream qualified as Stream
import Invar.Learn.Worker qualified as Worker
import Invar.Loop qualified as Loop
import Invar.Rollout qualified as Rollout
import Invar.Runtime qualified as Runtime
import Invar.Spec.Invocation qualified as V
import Invar.Store qualified as Store
import Invar.Worker qualified as Infer
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)

run :: [String] -> IO ()
run ["--help"] = putStrLn usage
run supplied = do
    fields <- either die pure (O.parse options supplied)
    case O.optional fields "resume" of
        Just directory
            | Map.size fields == 1 -> Runtime.resume directory declared taught >>= either (die . show) pure
            | otherwise -> die "--resume takes no other option; the run declaration supplies them"
        Nothing -> do
            (config, chosen) <- either die pure (routed fields)
            case (chosen, Learn.algorithm (Loop.settings config)) of
                (Concurrent lag, Learn.Grpo _) -> do
                    document <- Bytes.getContents >>= either die pure . Workload.decode
                    Runtime.run (Runtime.Run config lag document) (toJSON supplied) >>= either (die . show) pure
                (Concurrent lag, Learn.Opd _) -> do
                    teacher <- either die pure (teacherWorker fields)
                    document <- Bytes.getContents >>= either die pure . Workload.decodeOf @Cohort.Ungraded
                    Runtime.distill teacher (Runtime.Run config lag document) (toJSON supplied) >>= either (die . show) pure
                (Lockstep, Learn.Opd _) -> die "A distillation run goes through the event runtime, which shared execution enters with --staleness"
                (Lockstep, Learn.Grpo _) -> do
                    let selected = selection (Loop.settings config)
                    encoded <- Bytes.getContents
                    cycles <- either die pure (Dataset.decode selected encoded)
                    synchronous config selected cycles

data Route = Concurrent Natural | Lockstep

route :: Loop.Config -> Maybe Natural -> Route
route config declaredLag = case declaredLag of
    Just lag -> Concurrent lag
    Nothing
        | Loop.inferenceMode engine == Rollout.Shared || Loop.learningMode engine == Worker.Shared -> Lockstep
        | otherwise -> Concurrent 0
  where
    engine = Loop.backend config

routed :: O.Fields -> Either String (Loop.Config, Route)
routed fields = do
    config <- configure fields
    first show (Learn.validate (Loop.settings config))
    lag <- traverse (const (O.numeric fields "staleness")) (O.optional fields "staleness")
    pure (config, route config lag)

declared :: Value -> Either String (Loop.Config, Natural)
declared arguments = do
    fields <- parseEither parseJSON arguments >>= O.parse options
    when (isJust (O.optional fields "resume")) (Left "A run declaration cannot itself resume a run")
    (config, chosen) <- routed fields
    case chosen of
        Concurrent lag -> pure (config, lag)
        Lockstep -> Left "The run declaration declares shared execution without --staleness, which runs the lockstep loop and keeps no journal"

taught :: Value -> Either String Infer.Worker
taught arguments = parseEither parseJSON arguments >>= O.parse options >>= teacherWorker

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
    backend <- Loop.Backend <$> string "python" <*> string "inference-python" <*> string "inference" <*> pure (O.optional fields "inference-config") <*> InferenceInput.mode (O.optional fields "inference-mode") <*> string "learning" <*> pure (O.optional fields "learning-config") <*> learningMode (O.optional fields "learning-mode") <*> string "cache" <*> Dataset.sessions (O.optional fields "devices") <*> teacherMode (O.optional fields "teacher-mode")
    Loop.Config backend <$> string "output" <*> string "checkpoint" <*> string "reference" <*> (algorithm fields >>= (`settingsWith` fields)) <*> publication fields
  where
    string = O.required fields

algorithm :: O.Fields -> Either String Learn.Algorithm
algorithm fields = case O.optional fields "algorithm" of
    Just "opd" -> do
        when (isJust (O.optional fields "delta")) (Left "--delta applies to grpo")
        Learn.Opd <$> (Algorithm.Teacher <$> string "teacher-model" <*> string "teacher-revision" <*> string "teacher-digest" <*> string "tokenizer-digest" <*> string "teacher-base-digest" <*> string "teacher-assembly-digest" <*> O.numeric fields "teacher-temperature" <*> pure "none" <*> pure "processed" <*> pure "forced")
    selected
        | selected `notElem` [Nothing, Just "grpo"] -> Left "Invalid algorithm: expected grpo or opd"
        | any (isJust . O.optional fields) teacherNames -> Left "The teacher options apply to opd"
        | otherwise -> Learn.Grpo <$> O.numeric fields "delta"
  where
    string = O.required fields

teacherWorker :: O.Fields -> Either String Infer.Worker
teacherWorker fields = Infer.Worker <$> string "inference-python" <*> string "teacher" <*> string "cache" <*> string "teacher-adapter" <*> pure [] <*> pure (O.optional fields "teacher-config")
  where
    string = O.required fields

teacherMode :: Maybe String -> Either String Worker.Mode
teacherMode Nothing = Right Worker.Process
teacherMode (Just "process") = Right Worker.Process
teacherMode (Just "resident") = Right Worker.Resident
teacherMode _ = Left "Invalid teacher mode: expected process or resident"

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
settings fields = O.numeric fields "delta" >>= (`settingsWith` fields) . Learn.Grpo

settingsWith :: Learn.Algorithm -> O.Fields -> Either String Learn.Settings
settingsWith chosen fields = do
    optimizer <- Learn.Optimizer <$> number "rate" <*> number "beta1" <*> number "beta2" <*> number "optimizer-epsilon" <*> number "decay"
    current <- string "policy"
    Learn.Settings current <$> string "learner" <*> string "reference-digest" <*> string "tokenizer-digest" <*> string "base-digest" <*> string "assembly-digest" <*> string "behavior-base-digest" <*> string "behavior-assembly-digest" <*> number "clip" <*> number "penalty" <*> pure chosen <*> maybe (pure 1) (const (O.numeric fields "steps")) (O.optional fields "steps") <*> pure optimizer <*> pure (Learn.synchronous 0 current) <*> referenceSource fields
  where
    string = O.required fields
    number = O.numeric fields

referenceSource :: O.Fields -> Either String Learn.ReferenceSource
referenceSource fields = case O.optional fields "reference-source" of
    Nothing -> Right Learn.FromEngine
    Just named -> maybe (Left "Invalid reference source: expected engine or learner") Right (Stream.readSource (Text.pack named))

usage :: String
usage = usageInfo "Usage: invar train OPTIONS < tasks.json\nAll options except --devices, --inference-config, --inference-mode, --learning-mode, --steps, --staleness, --reference-source, --algorithm and the teacher options are required; opd takes the teacher options in place of --delta. Input is a nonempty JSON array of declared cycles.\nUsage: invar train --resume DIRECTORY\nResume an interrupted run of the event runtime from the journal in its output directory, which supplies every other option and the workload." options

options :: [OptDescr (String, String)]
options = O.descriptions [("publication", "Checkpoint publication: rename or reference"), ("devices", "Optional comma-separated CUDA devices, one rollout worker process per device"), ("python", "Learning Python executable"), ("inference-python", "Inference Python executable"), ("inference", "Inference worker script"), ("inference-config", "Optional inference worker launch configuration"), ("inference-mode", "Inference execution: serial (default), batch, resident or shared"), ("learning", "Update worker script"), ("learning-config", "Optional update worker configuration"), ("learning-mode", "Learning execution: process (default), resident or shared; shared requires both roles"), ("cache", "Pinned model cache"), ("output", "New output directory"), ("checkpoint", "Initial paired checkpoint directory"), ("reference", "Fixed reference adapter file"), ("staleness", "Optional staleness d, 0 when omitted: update u learns from rollouts of version max(0, u - d) while later rollouts may run. Every run goes through the event runtime except shared inference and learning without this option, which runs the lockstep loop; shared execution runs only with d = 0"), ("resume", "Output directory of an interrupted run of the event runtime; it takes no other option"), ("algorithm", "Learning algorithm: grpo (default), or opd, which reads a workload without answers or groups and needs every teacher option except --teacher-config and --teacher-mode")] ++ O.descriptions teacherOptions ++ settingsOptions

teacherOptions :: [(String, String)]
teacherOptions =
    [ ("teacher", "Distillation teacher score worker script, run with --inference-python and --cache")
    , ("teacher-adapter", "Teacher adapter file; a file with no tensors loads the teacher's base alone")
    , ("teacher-config", "Optional teacher worker configuration")
    , ("teacher-mode", "Teacher execution: process (default), one score process per rollout, or resident, one process for the run")
    , ("teacher-model", "Teacher model name")
    , ("teacher-revision", "Teacher model revision")
    , ("teacher-digest", "Teacher canonical adapter tensor SHA-256")
    , ("teacher-base-digest", "Teacher frozen model tensor SHA-256")
    , ("teacher-assembly-digest", "Teacher model assembly SHA-256")
    , ("teacher-temperature", "Temperature the teacher scores at; every task must be sampled at it")
    ]

teacherNames :: [String]
teacherNames = map fst teacherOptions

settingsOptions :: [OptDescr (String, String)]
settingsOptions = O.descriptions [("policy", "Consumed canonical policy tensor SHA-256"), ("tokenizer-digest", "Tokenizer operation SHA-256"), ("base-digest", "Learner frozen model tensor SHA-256"), ("assembly-digest", "Learner model assembly SHA-256"), ("behavior-base-digest", "Actual rollout frozen model SHA-256"), ("behavior-assembly-digest", "Actual rollout model assembly SHA-256"), ("learner", "Consumed learner file SHA-256"), ("reference-digest", "Canonical reference tensor SHA-256"), ("clip", "GRPO clipping coefficient"), ("penalty", "Reference penalty coefficient"), ("delta", "Advantage normalization epsilon"), ("steps", "Optional number of optimizer steps per update over consecutive mini-batches of the logical order (default 1)"), ("rate", "AdamW learning rate"), ("beta1", "AdamW first moment coefficient"), ("beta2", "AdamW second moment coefficient"), ("optimizer-epsilon", "AdamW epsilon"), ("decay", "AdamW weight decay"), ("reference-source", "Source of the objective's reference words: engine (default) or learner")]
