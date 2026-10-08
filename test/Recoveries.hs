{-# LANGUAGE OverloadedStrings #-}

module Recoveries (recoveries, inspections, ran, finishedRun, retriedRun) where

import BatchCalls (quote)
import Control.Monad (forM_, unless)
import Data.Aeson (Value (..), encode, object, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as Char
import Data.ByteString.Lazy qualified as Lazy
import Data.List (genericLength, isInfixOf)
import Hedgehog
import Invar.Async.Entry qualified as Entry
import Invar.Cohort qualified as C
import Invar.History.Runtime qualified as History
import Invar.Journal qualified as Journal
import Invar.Learn qualified as L
import Invar.Learn.Worker qualified as Learner
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
import Numeric.Natural (Natural)
import Sessions qualified
import Store (workspace)
import System.Directory (copyFile, createDirectory, doesDirectoryExist, getPermissions, listDirectory, removeDirectoryRecursive, removeFile, setOwnerWritable, setPermissions)
import System.Exit (ExitCode (ExitSuccess))
import System.FilePath ((</>))
import Workloads qualified

recoveries :: Group
recoveries =
    Group
        "Runtime run and resume"
        [ ("a finished run resumes from its own journal with every update committed and one restart entry", once finished)
        , ("every prefix of a run's journal, with the files a crash there leaves, is admitted by a resume", once prefixes)
        , ("a resume that cannot read a transcript or a generation, or that the replay refuses, leaves the journal's bytes unchanged", once refusals)
        , ("a crash after a process learner's reservation and before its attempt leaves a journal a resume admits", once interrupted)
        , ("a failure after the learner ran names the unresolved update and one before it does not", once reported)
        , ("a publication that would replace a generation already on disk is refused and leaves it untouched", once occupied)
        ]
  where
    once = withTests 1 . property

inspections :: Group
inspections =
    Group
        "Runtime history admission"
        [ ("a finished run is admitted from a read-only journal it leaves unchanged, with its own workload and checkpoint, its schedule, its committed attempt, the calls it consumed and the load of each process", once finishedHistory)
        , ("a run interrupted before its first publication and resumed with retry records is admitted with its concluded and its committed attempt", once retriedHistory)
        , ("a run whose publication a restart confirmed is admitted with that attempt committed", once confirmedHistory)
        , ("a process no protocol admitted contributes no load, even when its transcript holds one, and counts once among the processes", once unadmittedLoad)
        , ("an uncommitted update, a reservation without an end, a removed restart, a missing or extra generation and a declared workload the transcripts do not support are refused", once refusedHistories)
        ]
  where
    once = withTests 1 . property

data Fixture = Fixture {root :: FilePath, chosen :: Runtime.Run, inference :: FilePath}

prepared :: PropertyT IO Fixture
prepared = do
    base <- workspace
    sessions <- Sessions.options base 1
    let settings = F.configured
        definition = R.definition sessions
        cycle' = Loop.Cycle (C.tasks definition) (R.order sessions) (R.delivery sessions)
        members = fromIntegral (length (Loop.tasks cycle'))
    retried <- Sessions.optionsFrom base 1 (members + 1)
    initial <- evalEither (Policy.describe ("test-model", "test-revision") (L.policy settings, L.tokenizer settings, L.behaviorBase settings, L.behaviorAssembly settings))
    tasks <- evalEither (first show (Loop.bindTasks initial (Loop.tasks cycle')))
    evalIO $ do
        createDirectory (base </> "input")
        Policy.stageDescription (base </> "input" </> "policy.json") initial
    learners <- evalIO $ R.withConfiguredDriver R.Serial (R.worker sessions, R.sessions sessions) $ \driver -> do
        batch <- R.run driver sessions {R.definition = C.Definition (L.policy settings) tasks} >>= F.require
        planned <- F.require (L.prepare settings batch)
        first' <- F.prepare base (0, V.ordinal members) planned
        retry <- F.prepare base (1, V.ordinal (2 * members + 1)) planned
        pure (F.process (base </> "replies") first', F.process (base </> "replies") retry)
    (firstLearner, retryLearner) <- evalEither learners
    let copying index = unlines ["output=\"${4#--output=}\"", "mkdir -p \"$output\" || exit 24", unwords (["cp"] ++ [quote (base </> ("update" ++ show (index :: Int)) </> name) | name <- ["adapter.safetensors", "learner.pt", "gradients.safetensors", "probabilities.json"]] ++ ["\"$output\"", "|| exit 25"])]
        dispatching marker (firstScript, retryScript) = unlines ["if test -e " ++ quote marker ++ "; then exec /bin/sh " ++ quote retryScript ++ " \"$@\"; fi", ": > " ++ quote marker, "exec /bin/sh " ++ quote firstScript ++ " \"$@\""]
        worker = R.worker sessions
        inferring = base </> "inference.sh"
        backend = Loop.Backend "/bin/sh" (Worker.executable worker) inferring Nothing R.Serial (base </> "learner.sh") Learner.Process (Worker.cache worker) (R.sessions sessions)
        config = Loop.Config backend (base </> "run") (base </> "input") (base </> "reference") settings Store.RenameExclusive
    evalIO $ do
        writeFile (base </> "learner0.sh") (copying 0 ++ firstLearner)
        writeFile (base </> "learner1.sh") (copying 1 ++ retryLearner)
        writeFile (base </> "learner.sh") (dispatching (base </> "learned") (base </> "learner0.sh", base </> "learner1.sh"))
        writeFile inferring (dispatching (base </> "inferred") (Worker.script worker, Worker.script (R.worker retried)))
    document <- evalEither Sessions.workload
    pure (Fixture base (Runtime.Run config 0 document) inferring)

ran :: PropertyT IO Fixture
ran = do
    fixture <- prepared
    outcome <- evalIO (Runtime.run (chosen fixture) Null)
    either (\problem -> annotateShow problem >> failure) pure outcome
    pure fixture

interpreter :: Fixture -> FilePath -> Value -> Either String (Loop.Config, Natural)
interpreter fixture directory _ = case chosen fixture of
    Runtime.Run config lag _ -> Right (config {Loop.root = directory}, lag)

journaled :: FilePath -> IO [ByteString]
journaled directory = Char.lines <$> Bytes.readFile (directory </> "journal.jsonl")

decoded :: ByteString -> Either String Entry.Entry
decoded line = case Journal.entries (line <> "\n") of
    Right [fields] -> parseEither Entry.decode fields
    Right _ -> Left "Expected one entry"
    Left problem -> Left problem

copied :: FilePath -> FilePath -> IO ()
copied source target = do
    createDirectory target
    names <- listDirectory source
    forM_ names $ \name -> do
        directory <- doesDirectoryExist (source </> name)
        if directory then copied (source </> name) (target </> name) else copyFile (source </> name) (target </> name)

resumed :: Fixture -> FilePath -> IO (Either Runtime.Error ())
resumed fixture directory = Runtime.resume directory (interpreter fixture directory)

finished :: PropertyT IO ()
finished = do
    fixture <- ran
    let output = root fixture </> "run"
    before <- evalIO (journaled output)
    names <- evalIO (listDirectory output)
    outcome <- evalIO (resumed fixture output)
    either (\problem -> annotateShow problem >> failure) pure outcome
    after <- evalIO (journaled output)
    take (length before) after === before
    restart <- case drop (length before) after of
        [line] -> evalEither (decoded line)
        _ -> failure
    case restart of
        Entry.Restarted [generation] -> Entry.version generation === 1
        _ -> annotateShow restart >> failure
    evalIO (listDirectory output) >>= (=== names)

prefixes :: PropertyT IO ()
prefixes = do
    fixture <- ran
    let output = root fixture </> "run"
    complete <- evalIO (journaled output)
    entries <- traverse (evalEither . decoded) complete
    let recorded = length (takeWhile (not . recording) entries) + 1
        committed = length (takeWhile (not . committing) entries) + 1
    forM_ [1 .. length complete] $ \kept -> forM_ (published kept recorded committed) $ \generation -> do
        let copy = root fixture </> ("prefix" ++ show kept ++ (if generation then "published" else ""))
            reserved = [number | Entry.Reserved number _ _ _ <- take kept entries]
        evalIO $ do
            copied output copy
            Bytes.writeFile (copy </> "journal.jsonl") (Char.unlines (take kept complete) <> "{\"entry\":\"ev")
            files <- listDirectory (copy </> "transcripts")
            forM_ files $ \name -> unless (name `elem` [show number ++ ".jsonl" | number <- reserved]) (removeFile (copy </> "transcripts" </> name))
            unless generation (removeDirectoryRecursive (copy </> "generation1"))
        annotate ("kept " ++ show kept ++ (if generation then " with its generation" else ""))
        outcome <- evalIO (resumed fixture copy)
        case outcome of
            Left (Runtime.Recovery problem) -> annotate problem >> failure
            Left (Runtime.Declaration problem) -> annotate problem >> failure
            _ -> pure ()
        after <- evalIO (journaled copy)
        take kept after === take kept complete
        case drop kept after of
            line : _ ->
                evalEither (decoded line) >>= \entry -> case entry of
                    Entry.Restarted _ -> success
                    _ -> annotateShow entry >> failure
            [] -> failure
  where
    recording entry = entry == Entry.Happened (Entry.Recorded 0 0)
    committing entry = entry == Entry.Happened (Entry.Committed 0 0)
    published kept recorded committed
        | kept < recorded = [False]
        | kept < committed = [False, True]
        | otherwise = [True]

refusals :: PropertyT IO ()
refusals = do
    fixture <- ran
    let output = root fixture </> "run"
        refused copy change expected = do
            evalIO (copied output copy >> change copy >> Bytes.appendFile (copy </> "journal.jsonl") "{\"entry\":\"ev")
            unchanged <- evalIO (Bytes.readFile (copy </> "journal.jsonl"))
            outcome <- evalIO (resumed fixture copy)
            case outcome of
                Left (Runtime.Recovery problem) -> annotate problem >> assert (expected `isInfixOf` problem)
                _ -> annotateShow outcome >> failure
            evalIO (Bytes.readFile (copy </> "journal.jsonl")) >>= (=== unchanged)
    refused (root fixture </> "damaged") (\copy -> Bytes.appendFile (copy </> "generation1" </> "learner.pt") "\0") "Generation 1 differs from the update its attempt staged"
    refused (root fixture </> "missing") (\copy -> removeDirectoryRecursive (copy </> "generation1")) "Update 0 is committed in the journal but its generation is missing"
    refused (root fixture </> "unreadable") (appendLearner "not a record\n") "has no admitted result"
    refused (root fixture </> "older") older "another format"
    refused (root fixture </> "reserved") unreadable "cannot be read"
  where
    unreadable copy = do
        lines' <- journaled copy
        case [(index, number) | (index, Right (Entry.Reserved number Entry.Learner _ _)) <- zip [1 ..] (map decoded lines')] of
            (kept, number) : _ -> do
                Bytes.writeFile (copy </> "journal.jsonl") (Char.unlines (take kept lines'))
                let path = copy </> "transcripts" </> (show number ++ ".jsonl")
                removeFile path
                createDirectory path
            [] -> ioError (userError "The run reserved no learner process")
    appendLearner line copy = do
        entries <- journaled copy
        forM_ [number | Right (Entry.Reserved number Entry.Learner _ _) <- map decoded entries] $ \number ->
            Bytes.appendFile (copy </> "transcripts" </> (show number ++ ".jsonl")) line
    older copy = do
        lines' <- journaled copy
        case lines' of
            declaration : rest -> case Journal.entries (declaration <> "\n") of
                Right [fields] -> Bytes.writeFile (copy </> "journal.jsonl") (Char.unlines (Lazy.toStrict (encode (Object (Fields.insert "format" (String "invar-runtime-journal-v1") fields))) : rest))
                _ -> ioError (userError "The declaration is not one record")
            [] -> ioError (userError "The journal is empty")

interrupted :: PropertyT IO ()
interrupted = do
    fixture <- prepared
    let output = root fixture </> "run"
        transcribed = output </> "transcripts"
    evalIO (Bytes.readFile (inference fixture) >>= Bytes.writeFile (inference fixture) . (Char.pack ("chmod a-w " ++ quote transcribed ++ "\n") <>))
    outcome <- evalIO (Runtime.run (chosen fixture) Null)
    evalIO (getPermissions transcribed >>= setPermissions transcribed . setOwnerWritable True)
    case outcome of
        Left (Runtime.Crashed _) -> success
        _ -> annotateShow outcome >> failure
    complete <- evalIO (journaled output)
    entries <- traverse (evalEither . decoded) complete
    case reverse entries of
        Entry.Reserved _ Entry.Learner _ _ : _ -> success
        latest : _ -> annotateShow latest >> failure
        [] -> failure
    resumedOutcome <- evalIO (resumed fixture output)
    case resumedOutcome of
        Left (Runtime.Recovery problem) -> annotate problem >> failure
        Left (Runtime.Declaration problem) -> annotate problem >> failure
        _ -> pure ()
    after <- evalIO (journaled output)
    case drop (length complete) after of
        line : _ ->
            evalEither (decoded line) >>= \entry -> case entry of
                Entry.Restarted [] -> success
                _ -> annotateShow entry >> failure
        [] -> failure

reported :: PropertyT IO ()
reported = do
    engaged <- prepared
    evalIO (writeFile (root engaged </> "learner.sh") "exit 3\n")
    afterLearner <- evalIO (Runtime.run (chosen engaged) Null)
    case afterLearner of
        Left (Runtime.Unresolved 0 (Runtime.Learning _)) -> success
        unexpected -> annotateShow unexpected >> failure
    evalIO (doesDirectoryExist (root engaged </> "run" </> "generation1")) >>= (=== False)
    untouched <- prepared
    evalIO (writeFile (inference untouched) "exit 3\n")
    beforeLearner <- evalIO (Runtime.run (chosen untouched) Null)
    case beforeLearner of
        Left (Runtime.Rollout _) -> success
        unexpected -> annotateShow unexpected >> failure

occupied :: PropertyT IO ()
occupied = do
    fixture <- prepared
    let output = root fixture </> "run"
        generation = output </> "generation1"
    evalIO (Bytes.readFile (inference fixture) >>= Bytes.writeFile (inference fixture) . (Char.pack ("mkdir -p " ++ quote generation ++ "\n") <>))
    outcome <- evalIO (Runtime.run (chosen fixture) Null)
    case outcome of
        Left (Runtime.Unresolved 0 _) -> success
        unexpected -> annotateShow unexpected >> failure
    evalIO (listDirectory generation) >>= (=== [])

learnerInterval :: Entry.Entry -> Bool
learnerInterval (Entry.Elapsed role _ _ _) = role == "learner"
learnerInterval _ = False

membersOf :: Fixture -> Natural
membersOf fixture = case chosen fixture of
    Runtime.Run _ _ document -> sum [genericLength (Workload.tasks declared) | declared <- Workload.cycles document]

inspected :: Fixture -> FilePath -> PropertyT IO (Either String History.Checked)
inspected fixture directory = do
    evalIO (History.inspect directory (interpreter fixture directory))

attempts :: History.Checked -> PropertyT IO [Value]
attempts checked = evalEither (parseEither (withObject "history description" (.: "attempts")) (History.describe checked))

binding :: Natural -> Value
binding index = object ["call" .= index, "attempt" .= index, "instance" .= index]

interruptedBefore :: Fixture -> FilePath -> (Entry.Entry -> Bool) -> PropertyT IO ()
interruptedBefore fixture copy ending = do
    let output = root fixture </> "run"
    complete <- evalIO (journaled output)
    entries <- traverse (evalEither . decoded) complete
    let kept = length (takeWhile (not . ending) entries)
        reserved = [number | Entry.Reserved number _ _ _ <- take kept entries]
    evalIO $ do
        copied output copy
        Bytes.writeFile (copy </> "journal.jsonl") (Char.unlines (take kept complete) <> "{\"entry\":\"ev")
        files <- listDirectory (copy </> "transcripts")
        forM_ files $ \name -> unless (name `elem` [show number ++ ".jsonl" | number <- reserved]) (removeFile (copy </> "transcripts" </> name))

refusedHistory :: Fixture -> FilePath -> String -> PropertyT IO ()
refusedHistory fixture copy expected = do
    outcome <- inspected fixture copy
    case outcome of
        Left problem -> annotate problem >> assert (expected `isInfixOf` problem)
        Right _ -> failure

finishedHistory :: PropertyT IO ()
finishedHistory = do
    fixture <- ran
    let output = root fixture </> "run"
        members = membersOf fixture
        journal = output </> "journal.jsonl"
    evalIO (Bytes.appendFile journal "{\"entry\":\"ev")
    evalIO (getPermissions journal >>= setPermissions journal . setOwnerWritable False)
    before <- evalIO (Bytes.readFile journal)
    checked <- inspected fixture output >>= evalEither
    evalIO (Bytes.readFile journal) >>= (=== before)
    case chosen fixture of
        Runtime.Run _ _ document -> History.workload checked === document
    Loop.checkpoint (History.config checked) === root fixture </> "input"
    length [() | Object fields <- History.loads checked, Fields.lookup "stage" fields == Just "load"] === 2
    length (History.generations checked) === 1
    History.identities checked === members + 1
    recorded <- attempts checked
    recorded === [object ["update" .= (0 :: Int), "attempt" .= (0 :: Int), "binding" .= binding members, "process" .= (1 :: Int), "consumed" .= map binding [0 .. members - 1], "outcome" .= object ["committed" .= (1 :: Int)]]]

finishedRun :: Fixture -> PropertyT IO History.Checked
finishedRun fixture = inspected fixture (root fixture </> "run") >>= evalEither

retriedRun :: Fixture -> PropertyT IO History.Checked
retriedRun fixture = do
    let copy = root fixture </> "retried"
    interruptedBefore fixture copy learnerInterval
    evalIO (removeDirectoryRecursive (copy </> "generation1"))
    outcome <- evalIO (resumed fixture copy)
    either (\problem -> annotateShow problem >> failure) pure outcome
    inspected fixture copy >>= evalEither

retriedHistory :: PropertyT IO ()
retriedHistory = do
    fixture <- ran
    let members = membersOf fixture
    checked <- retriedRun fixture
    recorded <- attempts checked
    map (parseEither (withObject "attempt" (\fields -> (,) <$> fields .: "attempt" <*> fields .: "outcome"))) recorded === [Right (0 :: Int, object ["concluded" .= True]), Right (1, object ["committed" .= (1 :: Int)])]
    map (parseEither (withObject "attempt" (.: "consumed"))) recorded === [Right (map binding [0 .. members - 1]), Right (map binding [members + 1 .. 2 * members])]
    History.identities checked === 2 * members + 2

confirmedHistory :: PropertyT IO ()
confirmedHistory = do
    fixture <- ran
    let copy = root fixture </> "confirmed"
    interruptedBefore fixture copy (== Entry.Happened (Entry.Committed 0 0))
    outcome <- evalIO (resumed fixture copy)
    either (\problem -> annotateShow problem >> failure) pure outcome
    checked <- inspected fixture copy >>= evalEither
    recorded <- attempts checked
    map (parseEither (withObject "attempt" (.: "outcome"))) recorded === [Right (object ["committed" .= (1 :: Int)])]

unadmittedLoad :: PropertyT IO ()
unadmittedLoad = do
    fixture <- ran
    let output = root fixture </> "run"
        copy = root fixture </> "idle"
        idle = [Entry.Reserved 99 Entry.Inference 0 0, Entry.Finished 99 (Transcript.Exited ExitSuccess Transcript.Complete)]
    evalIO $ do
        copied output copy
        copyFile (copy </> "transcripts" </> "0.jsonl") (copy </> "transcripts" </> "99.jsonl")
        Bytes.appendFile (copy </> "journal.jsonl") (Char.unlines (map (Lazy.toStrict . encode . Entry.encode) idle))
    checked <- inspected fixture copy >>= evalEither
    length [() | Object fields <- History.loads checked, Fields.lookup "stage" fields == Just "load"] === 2
    parseEither (withObject "recorded execution" (.: "processes")) (History.recorded checked) === Right (3 :: Int)

refusedHistories :: PropertyT IO ()
refusedHistories = do
    fixture <- ran
    let output = root fixture </> "run"
        refusedWith copy (change :: FilePath -> IO ()) expected = do
            evalIO (copied output copy >> change copy)
            refusedHistory fixture copy expected
    refusedWith (root fixture </> "unended") (\copy -> Bytes.appendFile (copy </> "journal.jsonl") (Lazy.toStrict (encode (Entry.encode (Entry.Reserved 99 Entry.Inference 0 0))) <> "\n")) "no recorded end"
    refusedWith (root fixture </> "missinggeneration") (\copy -> removeDirectoryRecursive (copy </> "generation1")) "published generations differ"
    refusedWith (root fixture </> "uncommitted") (\copy -> journaled copy >>= \lines' -> Bytes.writeFile (copy </> "journal.jsonl") (Char.unlines (takeWhile ((/= Right (Entry.Happened (Entry.Committed 0 0))) . decoded) lines'))) "not committed every declared update"
    refusedWith (root fixture </> "extrageneration") (\copy -> copied (copy </> "generation1") (copy </> "generation2")) "published generations differ"
    refusedWith (root fixture </> "otherworkload") (\copy -> journaled copy >>= \lines' -> Bytes.writeFile (copy </> "journal.jsonl") (Char.unlines (map (Workloads.everywhere "\"seed\":17" "\"seed\":18") (take 1 lines') ++ drop 1 lines'))) "differs from the declared invocation"
    let copy = root fixture </> "unrestarted"
    interruptedBefore fixture copy learnerInterval
    evalIO (removeDirectoryRecursive (copy </> "generation1"))
    resumedOutcome <- evalIO (resumed fixture copy)
    either (\problem -> annotateShow problem >> failure) pure resumedOutcome
    evalIO $ do
        lines' <- journaled copy
        Bytes.writeFile (copy </> "journal.jsonl") (Char.unlines [line | line <- lines', either (const True) (not . restarting) (decoded line)])
    refusedHistory fixture copy "(Epoch 1)"
  where
    restarting Entry.Restarted {} = True
    restarting _ = False
