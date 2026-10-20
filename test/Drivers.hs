{-# LANGUAGE OverloadedStrings #-}

module Drivers (drivers) where

import BatchCalls (quote)
import Control.Monad (forM, forM_)
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
import Invar.Learn.Report qualified as Report
import Invar.Learn.Worker qualified as Learner
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
import System.Directory (createDirectory)
import System.FilePath ((</>))

drivers :: Group
drivers =
    Group
        "Runtime publication"
        [("the runtime at staleness zero publishes the learner's generations over two cycles in which the policy changes and the second rollout scores the reference", withTests 1 (property agreed))]

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
        backend = Loop.Backend "/bin/sh" (Worker.executable worker) inferring Nothing R.Serial (base </> "learner.sh") Nothing Learner.Process (Worker.cache worker) (R.sessions opening) Learner.Process
        config output = Loop.Config backend output (base </> "input") (base </> "reference") settings Store.RenameExclusive
        updated = settings {L.policy = adapter, L.learner = learner, L.schedule = L.synchronous 1 adapter}
        scoring = Loop.scoring (config base) (Loop.Checkpoint "" adapter learner)
    secondLearner <- learned preparing (base, 1, V.ordinal (2 * members + 1)) (updated, successor, scoring)
    let copying index = unlines ["output=\"${4#--output=}\"", "mkdir -p \"$output\" || exit 24", unwords (["cp"] ++ [quote (base </> ("update" ++ show (index :: Int)) </> name) | name <- ["adapter.safetensors", "learner.pt", "gradients.safetensors", "probabilities.json"]] ++ ["\"$output\"", "|| exit 25"])]
        dispatching marker (firstScript, laterScript) = unlines ["if test -e " ++ quote marker ++ "; then exec /bin/sh " ++ quote laterScript ++ " \"$@\"; fi", ": > " ++ quote marker, "exec /bin/sh " ++ quote firstScript ++ " \"$@\""]
    evalIO $ do
        writeFile (base </> "learner0.sh") (copying 0 ++ firstLearner)
        writeFile (base </> "learner1.sh") (copying 1 ++ secondLearner)
        writeFile (base </> "learner.sh") (dispatching (base </> "learned") (base </> "learner0.sh", base </> "learner1.sh"))
        writeFile inferring (dispatching (base </> "inferred") (Worker.script worker, Worker.script (R.worker following)))
    document <- evalEither (Sessions.workloads 2)
    outcome <- evalIO (Runtime.run (Runtime.Run (config (base </> "runtime")) 0 document) Null)
    either (\problem -> annotateShow problem >> failure) pure outcome
    checked <- evalIO (History.inspect (base </> "runtime") (const (Right (config (base </> "runtime"), 0)))) >>= evalEither
    let runtimeReports = [Cohort.update (Generation.cohort generation) | generation <- History.generations checked]
    length runtimeReports === 2
    L.policy settings /== adapter
    flags <- evalEither (traverse (parseEither scored . Report.request) runtimeReports)
    concat flags === [False, True]
    second <- evalIO (Policy.identity (base </> "update1" </> "adapter.safetensors"))
    adapter /== second
    later <- evalEither (Policy.successor second successor)
    forM_ (zip3 ["generation1", "generation2"] ["update0", "update1"] [successor, later]) $ \(published, update, described) -> do
        forM_ ["adapter.safetensors", "learner.pt"] $ \name -> do
            expected <- evalIO (Bytes.readFile (base </> update </> name))
            actual <- evalIO (Bytes.readFile (base </> "runtime" </> published </> name))
            expected === actual
        evalIO (Policy.readDescription (base </> "runtime" </> published </> "policy.json")) >>= (=== described)
  where
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
