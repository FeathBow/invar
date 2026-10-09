{-# LANGUAGE OverloadedStrings #-}

module Drivers (drivers) where

import BatchCalls (quote)
import Control.Monad (forM, forM_, when)
import Data.Aeson (Value (..), withObject, (.:))
import Data.Aeson.Types (Parser, parseEither)
import Data.Bifunctor (first)
import Data.ByteString qualified as Bytes
import Hedgehog
import Invar.Artifact qualified as Artifact
import Invar.Cohort qualified as C
import Invar.History.Cohort qualified as Cohort
import Invar.History.Generation qualified as Generation
import Invar.History.Runtime qualified as History
import Invar.Learn qualified as L
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Report qualified as Report
import Invar.Learn.Worker qualified as Learner
import Invar.Loop qualified as Loop
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as R
import Invar.Runtime qualified as Runtime
import Invar.Spec.Invocation qualified as V
import Invar.Store qualified as Store
import Invar.Worker qualified as Worker
import Invar.Workload qualified as Workload
import LearnerFixture qualified as F
import Sessions qualified
import Store (workspace)
import System.Directory (createDirectory, doesFileExist, removeFile)
import System.FilePath ((</>))

drivers :: Group
drivers =
    Group
        "Lockstep and runtime drivers"
        [("the lockstep driver and the runtime at staleness zero publish the same generations from the same learner inputs over two cycles in which the policy changes and the second rollout scores the reference", withTests 1 (property agreed))]

agreed :: PropertyT IO ()
agreed = do
    base <- workspace
    let settings = F.configured
    opening <- Sessions.options base 1
    let members = fromIntegral (length (C.tasks (R.definition opening)))
    initial <- evalEither (Policy.describe ("test-model", "test-revision") (L.policy settings, L.tokenizer settings, L.behaviorBase settings, L.behaviorAssembly settings))
    evalIO $ do
        createDirectory (base </> "input")
        Policy.stageDescription (base </> "input" </> "policy.json") initial
    firstLearner <- learned opening (base, 0, V.ordinal members) (settings, initial, Nothing)
    adapter <- evalIO (Policy.identity (base </> "update0" </> "adapter.safetensors"))
    learner <- evalIO (Artifact.identity "Learner checkpoint" (base </> "update0" </> "learner.pt"))
    successor <- evalEither (Policy.successor adapter initial)
    following <- Sessions.optionsWith base 1 (members + 1) (adapter, Just (L.reference settings))
    evalIO (createDirectory (base </> "prepared"))
    preparing <- Sessions.optionsWith (base </> "prepared") 1 0 (adapter, Just (L.reference settings))
    let worker = R.worker opening
        inferring = base </> "inference.sh"
        backend = Loop.Backend "/bin/sh" (Worker.executable worker) inferring Nothing R.Serial (base </> "learner.sh") Learner.Process (Worker.cache worker) (R.sessions opening)
        config output = Loop.Config backend output (base </> "input") (base </> "reference") settings Store.RenameExclusive
        updated = settings {L.policy = adapter, L.learner = learner, L.schedule = L.synchronous 1 adapter}
        scoring = Loop.scoring (config base) (Loop.Checkpoint "" adapter learner)
    secondLearner <- learned preparing (base, 1, V.ordinal (2 * members + 1)) (updated, successor, scoring)
    let copying index = unlines ["output=\"${4#--output=}\"", "mkdir -p \"$output\" || exit 24", unwords (["cp"] ++ [quote (base </> ("update" ++ show (index :: Int)) </> name) | name <- ["adapter.safetensors", "learner.pt", "gradients.safetensors", "probabilities.json"]] ++ ["\"$output\"", "|| exit 25"])]
        dispatching marker (firstScript, laterScript) = unlines ["if test -e " ++ quote marker ++ "; then exec /bin/sh " ++ quote laterScript ++ " \"$@\"; fi", ": > " ++ quote marker, "exec /bin/sh " ++ quote firstScript ++ " \"$@\""]
        markers = [base </> "learned", base </> "inferred"]
        reset = forM_ markers $ \marker -> doesFileExist marker >>= (`when` removeFile marker)
    evalIO $ do
        writeFile (base </> "learner0.sh") (copying 0 ++ firstLearner)
        writeFile (base </> "learner1.sh") (copying 1 ++ secondLearner)
        writeFile (base </> "learner.sh") (dispatching (base </> "learned") (base </> "learner0.sh", base </> "learner1.sh"))
        writeFile inferring (dispatching (base </> "inferred") (Worker.script worker, Worker.script (R.worker following)))
    document <- evalEither (Sessions.workloads 2)
    let instantiate policy = Loop.instantiate (policy, L.tokenizer settings, L.behaviorBase settings, L.behaviorAssembly settings)
    evalIO reset
    lockstep <- evalIO $ Loop.withDriver (config (base </> "lockstep")) $ \driver -> do
        let advance policy workload = either (pure . Left . Loop.Policy) (Loop.run driver) (instantiate policy workload)
        case Workload.cycles document of
            [cycle1, cycle2] -> do
                firstGeneration <- advance (L.policy settings) cycle1
                case firstGeneration of
                    Left problem -> pure (Left problem)
                    Right generated -> fmap (\later -> map observed [generated, later]) <$> advance (Loop.policy (Loop.current generated)) cycle2
            _ -> pure (Left (Loop.Policy "Expected two cycles"))
    generations <- evalEither (first show lockstep) >>= evalEither . first show
    evalIO reset
    outcome <- evalIO (Runtime.run (Runtime.Run (config (base </> "runtime")) 0 document) Null)
    either (\problem -> annotateShow problem >> failure) pure outcome
    checked <- evalIO (History.inspect (base </> "runtime") (const (Right (config (base </> "runtime"), 0)))) >>= evalEither
    let runtimeReports = [Cohort.update (Generation.cohort generation) | generation <- History.generations checked]
    map fst generations === map Report.checkedRequest runtimeReports
    L.policy settings /== adapter
    map snd generations === [adapter, adapter]
    flags <- evalEither (traverse (parseEither scored . Report.request) runtimeReports)
    concat flags === [False, True]
    forM_ ["generation1", "generation2"] $ \published -> forM_ ["policy.json", "adapter.safetensors", "learner.pt"] $ \name -> do
        left <- evalIO (Bytes.readFile (base </> "lockstep" </> published </> name))
        right <- evalIO (Bytes.readFile (base </> "runtime" </> published </> name))
        left === right
  where
    observed generated = (P.checkedRequest (Loop.result generated), Loop.policy (Loop.current generated))
    scored :: Value -> Parser [Bool]
    scored = withObject "update request" $ \fields -> do
        samples <- fields .: "samples"
        bits <- forM samples (withObject "sample" (.: "reference_bits")) :: Parser [[Value]]
        pure [not (all null bits)]
    learned options (root, index, binding) (chosen, described, reference) = do
        bound <- evalEither (first show (Loop.bindTasks described (C.tasks (R.definition options))))
        prepared <- evalIO $ R.withConfiguredDriver R.Serial (R.worker options, R.sessions options) $ \driver -> do
            batch <- R.run driver options {R.definition = C.Definition (L.policy chosen) bound, R.reference = reference} >>= F.require
            planned <- F.require (L.prepare chosen batch)
            F.process (root </> "replies") <$> F.prepare root (index, binding) planned
        evalEither prepared
