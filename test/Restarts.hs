{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

module Restarts (restarts) where

import BatchCalls (quote, session)
import Control.Monad (forM, forM_, unless)
import Data.Aeson (Value (..), encode, object, withObject, (.:), (.=))
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as Char
import Data.ByteString.Lazy qualified as Lazy
import Data.List (isInfixOf)
import Hedgehog
import Invar.Artifact qualified as Artifact
import Invar.Async.Entry qualified as Entry
import Invar.Cohort qualified as C
import Invar.History.Cohort qualified as Cohort
import Invar.History.Generation qualified as Generation
import Invar.History.Runtime qualified as History
import Invar.Infer.Invocation qualified as Invocation
import Invar.Journal qualified as Journal
import Invar.Learn qualified as L
import Invar.Learn.Report qualified as Report
import Invar.Learn.Worker qualified as Learner
import Invar.Loop qualified as Loop
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as R
import Invar.Runtime qualified as Runtime
import Invar.Spec.Invocation qualified as V
import Invar.Store qualified as Store
import LearnerFixture qualified as F
import Numeric.Natural (Natural)
import Sessions qualified
import Store (workspace)
import System.Directory (copyFile, createDirectory, doesDirectoryExist, listDirectory, removeDirectoryRecursive, removeFile)
import System.FilePath ((</>))

restarts :: Group
restarts =
    Group
        "Runtime histories across updates and restarts"
        [ ("a run at staleness one is admitted with both updates learning from version zero under the declared schedule", once stale)
        , ("a run resumed after a lost publication acknowledgement and again after a lost learner receipt is admitted with its failed attempt, each update committed once and each committed attempt using its own update's requests", once windows)
        , ("a dispatch before the commit that allows it, a commit before its record, an epoch taken back after a restart and a retry under another attempt number are refused", once mutated)
        ]
  where
    once = withTests 1 . property

data Fixture = Fixture {base :: FilePath, chosen :: Runtime.Run}

-- Two updates of the session workload. The workers answer by the request they receive, so a rollout or an update can run under each binding the run may assign it: at a positive staleness the learner of update 0 and the rollout of update 1 reserve their bindings concurrently, and a resumed run assigns new ones.
prepared :: Natural -> ([Natural], [Natural]) -> ([Natural], [Natural]) -> PropertyT IO Fixture
prepared lag (opening, later) (learning, following) = do
    root <- workspace
    let settings = F.configured
        initial = L.policy settings
        backend = Loop.Backend "/bin/sh" "/bin/sh" (root </> "inference.sh") Nothing R.Serial (root </> "learner.sh") Nothing Learner.Process root [[]] Learner.Process
        config = Loop.Config backend (root </> "run") (root </> "input") (root </> "reference") settings Store.RenameExclusive
    described <- evalEither (Policy.describe ("test-model", "test-revision") (initial, L.tokenizer settings, L.behaviorBase settings, L.behaviorAssembly settings))
    evalIO (createDirectory (root </> "input") >> Policy.stageDescription (root </> "input" </> "policy.json") described)
    updating <- learners root config (0, 0) (settings {L.schedule = L.Schedule 0 lag 0 initial}, described) learning
    adapter <- evalIO (Policy.identity (root </> "update0" </> "adapter.safetensors"))
    learner <- evalIO (Artifact.identity "Learner checkpoint" (root </> "update0" </> "learner.pt"))
    successor <- evalEither (Policy.successor adapter described)
    let (version, behavior, behaving) = if lag == 0 then (1, adapter, successor) else (0, initial, described)
    continuing <- learners root config (1, length learning) (settings {L.policy = adapter, L.learner = learner, L.schedule = L.Schedule 1 lag version behavior}, behaving) following
    inferring <- rollouts root settings ([(offset, initial) | offset <- opening] ++ [(offset, behavior) | offset <- later])
    evalIO $ do
        writeFile (root </> "inference.sh") inferring
        writeFile (root </> "learner.sh") (unlines (["IFS= read -r input || exit 21", "if false; then :"] ++ updating ++ continuing ++ ["fi", "exit 26"]))
    document <- evalEither (Sessions.workloads 2)
    pure (Fixture root (Runtime.Run config lag document))

-- One branch per binding: each copies its own prepared artifacts and replays the learner records prepared for that binding.
learners :: FilePath -> Loop.Config -> (Natural, Int) -> (L.Settings, Policy.Description) -> [Natural] -> PropertyT IO [String]
learners root config (update, numbered) (settings, described) bindings = do
    let behavior = L.behaviorPolicy (L.schedule settings)
        directory = root </> ("prepared" ++ show update)
    evalIO (createDirectory directory)
    options <- Sessions.optionsWith directory 1 0 (behavior, if behavior == L.reference settings then Nothing else Just (L.reference settings))
    bound <- evalEither (first show (Loop.bindTasks described (C.tasks (R.definition options))))
    branches <- evalIO $ R.withConfiguredDriver R.Serial (R.worker options, R.sessions options) $ \driver -> do
        batch <- R.run driver options {R.definition = C.Definition behavior bound, R.reference = Loop.scoring config (Loop.Checkpoint "" behavior (L.learner settings))} >>= F.require
        planned <- F.require (L.prepare settings batch)
        forM (zip [numbered ..] bindings) $ \(index, ordinal) -> branch index <$> F.prepare root (index, V.ordinal ordinal) planned
    concat <$> evalEither branches
  where
    branch index exchange =
        ["elif test \"$input\" = " ++ quote (Learner.input (F.call exchange)) ++ "; then", "output=\"${4#--output=}\"", "mkdir -p \"$output\" || exit 24", unwords (["cp"] ++ [quote (root </> ("update" ++ show (index :: Int)) </> name) | name <- ["adapter.safetensors", "learner.pt", "gradients.safetensors", "probabilities.json"]] ++ ["\"$output\"", "|| exit 25"])]
            ++ drop 1 (lines (F.process (root </> "replies") exchange))
            ++ ["exit 0"]

-- One branch per first binding of a rollout of one session.
rollouts :: FilePath -> L.Settings -> [(Natural, String)] -> PropertyT IO String
rollouts root settings variants = do
    branches <- forM variants $ \(offset, policy) -> do
        (calls, _) <- Sessions.setup 1 offset (policy, if policy == L.reference settings then Nothing else Just (L.reference settings))
        case calls of
            (call, _) : _ -> pure (("elif test \"$request\" = " ++ quote (Char.unpack (Invocation.batchInput call)) ++ "; then") : drop 1 (lines (session root calls)) ++ ["exit 0"])
            [] -> failure
    pure (unlines (["IFS= read -r request || exit 21", "if false; then :"] ++ concat branches ++ ["fi", "exit 31"]))

interpreter :: Fixture -> FilePath -> Value -> Either String (Loop.Config, Natural)
interpreter fixture directory _ = case chosen fixture of
    Runtime.Run config lag _ -> Right (config {Loop.root = directory}, lag)

ran :: Fixture -> PropertyT IO FilePath
ran fixture = do
    outcome <- evalIO (Runtime.run (chosen fixture) Null)
    either (\problem -> annotateShow problem >> failure) pure outcome
    pure (base fixture </> "run")

resumed :: Fixture -> FilePath -> PropertyT IO ()
resumed fixture directory = do
    outcome <- evalIO (Runtime.resume directory (interpreter fixture directory) (const (Left "A GRPO run has no teacher")))
    either (\problem -> annotateShow problem >> failure) pure outcome

inspected :: Fixture -> FilePath -> PropertyT IO (Either String History.Checked)
inspected fixture directory = evalIO (History.inspect directory (interpreter fixture directory))

entries :: FilePath -> PropertyT IO [Entry.Entry]
entries directory = do
    recorded <- evalIO (Bytes.readFile (directory </> "journal.jsonl")) >>= evalEither . Journal.entries
    evalEither (traverse (parseEither Entry.decode) recorded)

written :: FilePath -> [Entry.Entry] -> IO ()
written directory changed = Bytes.writeFile (directory </> "journal.jsonl") (Char.unlines (map (Lazy.toStrict . encode . Entry.encode) changed))

copied :: FilePath -> FilePath -> IO ()
copied source target = do
    createDirectory target
    names <- listDirectory source
    forM_ names $ \name -> do
        directory <- doesDirectoryExist (source </> name)
        if directory then copied (source </> name) (target </> name) else copyFile (source </> name) (target </> name)

-- A copy of the run as a crash just before the first entry that ends matches would leave it: the later entries, the transcripts of processes reserved after that point and the generations published after it are gone.
interrupted :: FilePath -> FilePath -> (Entry.Entry -> Bool) -> [String] -> PropertyT IO ()
interrupted source copy ending removed = do
    complete <- entries source
    let kept = takeWhile (not . ending) complete
        reserved = [show number ++ ".jsonl" | Entry.Reserved number _ _ _ <- kept]
    evalIO $ do
        copied source copy
        written copy kept
        files <- listDirectory (copy </> "transcripts")
        forM_ files $ \name -> unless (name `elem` reserved) (removeFile (copy </> "transcripts" </> name))
        forM_ removed $ \name -> removeDirectoryRecursive (copy </> name)

-- The run, then the same run resumed after its publication of update 0 lost its acknowledgement, and that resumed run resumed again after its learner of update 1 ran without its receipt arriving.
resumedTwice :: Fixture -> PropertyT IO (FilePath, FilePath)
resumedTwice fixture = do
    output <- ran fixture
    let confirmed = base fixture </> "confirmed"
        retried = base fixture </> "retried"
    interrupted output confirmed (== Entry.Happened (Entry.Committed 0 0)) ["generation2"]
    resumed fixture confirmed
    interrupted confirmed retried learnedSecond ["generation2"]
    resumed fixture retried
    pure (output, retried)
  where
    learnedSecond (Entry.Elapsed role update _ _) = role == "learner" && update == 1
    learnedSecond _ = False

stale :: PropertyT IO ()
stale = do
    fixture <- prepared 1 ([0], [5, 6]) ([5, 10], [11])
    output <- ran fixture
    checked <- inspected fixture output >>= evalEither
    let history = History.describe checked
    parseEither (withObject "history" (.: "staleness")) history === Right (1 :: Natural)
    parseEither (withObject "history" (\fields -> fields .: "updates" >>= traverse (withObject "update" (.: "version")))) history === Right [0, 0 :: Natural]
    traverse (parseEither scheduled . Report.request . Cohort.update . Generation.cohort) (History.generations checked) === Right [(0, 1, replicate 5 0), (1, 1, replicate 5 0)]
  where
    scheduled = withObject "update request" $ \fields -> do
        (update, lag) <- fields .: "schedule" >>= withObject "schedule" (\schedule -> (,) <$> schedule .: "update" <*> schedule .: "staleness")
        versions <- fields .: "samples" >>= traverse (withObject "sample" (.: "version"))
        pure (update :: Natural, lag :: Natural, versions :: [Natural])

windows :: PropertyT IO ()
windows = do
    fixture <- prepared 0 ([0], [6, 12]) ([5], [11, 17])
    (_, retried) <- resumedTwice fixture
    checked <- inspected fixture retried >>= evalEither
    let history = History.describe checked
        attempt = withObject "attempt" $ \fields -> (,,,) <$> fields .: "update" <*> fields .: "attempt" <*> fields .: "consumed" <*> fields .: "outcome"
    recorded <- evalEither (parseEither (withObject "history" (\fields -> fields .: "attempts" >>= traverse attempt)) history)
    [(update, tried, outcome) | (update, tried, _ :: [Value], outcome :: Value) <- recorded] === [(0 :: Natural, 0 :: Natural, committed 1), (1, 1, concluded), (1, 2, committed 2)]
    [traverse callOf consumed | (_, _, consumed, _) <- recorded] === [Right [0 .. 4], Right [6 .. 10], Right [12 .. 16]]
    parseEither (withObject "history" (.: "restarts")) (History.recorded checked) === Right (2 :: Natural)
  where
    committed version = object ["committed" .= (version :: Natural)]
    concluded = object ["concluded" .= True]
    callOf = parseEither (withObject "binding" (.: "call")) :: Value -> Either String Natural

mutated :: PropertyT IO ()
mutated = do
    fixture <- prepared 0 ([0], [6, 12]) ([5], [11, 17])
    (output, retried) <- resumedTwice fixture
    _ <- inspected fixture output >>= evalEither
    _ <- inspected fixture retried >>= evalEither
    let refused source name change expected = do
            let copy = base fixture </> name
            complete <- entries source
            evalIO (copied source copy >> written copy (change complete))
            outcome <- inspected fixture copy
            case outcome of
                Left problem -> annotate problem >> assert (expected `isInfixOf` problem)
                Right _ -> annotate name >> failure
    refused output "early" early "A request is dispatched without the core's dispatch command"
    refused output "unrecorded" (swapped (Entry.Happened (Entry.Recorded 1 1)) (Entry.Happened (Entry.Committed 1 1))) "Unexpected (Committed"
    refused retried "reverted" reverted "StaleEpoch"
    refused retried "renumbered" (map renumbered) "An attempt is journaled under a number the core did not assign"
  where
    -- The second rollout's process and its dispatches move ahead of the commit of update 0, the event that let the core dispatch them.
    early complete =
        let moving = case [number | Entry.Dispatched 5 _ _ _ number <- complete] of
                number : _ -> \case
                    Entry.Reserved owner _ _ _ -> owner == number
                    Entry.Dispatched _ _ _ _ owner -> owner == number
                    _ -> False
                [] -> const False
            (before, after) = break (== Entry.Happened (Entry.Committed 0 0)) (filter (not . moving) complete)
         in before ++ filter moving complete ++ after
    swapped one other = map (\recorded -> if recorded == one then other else if recorded == other then one else recorded)
    -- The segment after the first restart claims epoch 0 again, the epoch the first segment used.
    reverted complete = case break restarting complete of
        (before, restart : rest) -> let (segment, later) = break restarting rest in before ++ restart : map back segment ++ later
        _ -> complete
    restarting Entry.Restarted {} = True
    restarting _ = False
    back recorded = case recorded of
        Entry.Opened worker 1 -> Entry.Opened worker 0
        Entry.Reserved number role slot 1 -> Entry.Reserved number role slot 0
        Entry.Dispatched request worker 1 bound number -> Entry.Dispatched request worker 0 bound number
        Entry.Happened (Entry.Connected worker 1) -> Entry.Happened (Entry.Connected worker 0)
        Entry.Happened (Entry.Started worker 1 request) -> Entry.Happened (Entry.Started worker 0 request)
        Entry.Happened (Entry.Completed worker 1 request digest) -> Entry.Happened (Entry.Completed worker 0 request digest)
        _ -> recorded
    renumbered recorded = case recorded of
        Entry.Attempted 1 2 bound number -> Entry.Attempted 1 3 bound number
        _ -> recorded
