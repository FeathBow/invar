{-# LANGUAGE OverloadedStrings #-}

module Recoveries (recoveries) where

import BatchCalls (quote)
import Control.Monad (forM_, unless)
import Data.Aeson (Value (..), encode)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as Char
import Data.ByteString.Lazy qualified as Lazy
import Data.List (isInfixOf)
import Hedgehog
import Invar.Async.Entry qualified as Entry
import Invar.Cohort qualified as C
import Invar.Journal qualified as Journal
import Invar.Learn qualified as L
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
import System.Directory (copyFile, createDirectory, doesDirectoryExist, getPermissions, listDirectory, removeDirectoryRecursive, removeFile, setOwnerWritable, setPermissions)
import System.FilePath ((</>))

recoveries :: Group
recoveries =
    Group
        "Runtime run and resume"
        [ ("a finished run resumes from its own journal with every update committed and one restart entry", once finished)
        , ("every prefix of a run's journal, with the files a crash there leaves, is admitted by a resume", once prefixes)
        , ("a resume that cannot read a transcript or a generation, or that the replay refuses, leaves the journal's bytes unchanged", once refusals)
        , ("a crash after a process learner's reservation and before its attempt leaves a journal a resume admits", once interrupted)
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
    initial <- evalEither (Policy.describe ("test-model", "test-revision") (L.policy settings, L.tokenizer settings, L.behaviorBase settings, L.behaviorAssembly settings))
    tasks <- evalEither (first show (Loop.bindTasks initial (Loop.tasks cycle')))
    evalIO $ do
        createDirectory (base </> "input")
        Policy.stageDescription (base </> "input" </> "policy.json") initial
    learner <- evalIO $ R.withConfiguredDriver R.Serial (R.worker sessions, R.sessions sessions) $ \driver -> do
        batch <- R.run driver sessions {R.definition = C.Definition (L.policy settings) tasks} >>= F.require
        planned <- F.require (L.prepare settings batch)
        exchange <- F.prepare base (0, V.ordinal members) planned
        pure (F.process (base </> "replies") exchange)
    script <- evalEither learner
    let artifacts = base </> "update0"
        copying = unlines ["output=\"${4#--output=}\"", "mkdir -p \"$output\" || exit 24", unwords (["cp"] ++ [quote (artifacts </> name) | name <- ["adapter.safetensors", "learner.pt", "gradients.safetensors", "probabilities.json"]] ++ ["\"$output\"", "|| exit 25"])]
        worker = R.worker sessions
        backend = Loop.Backend "/bin/sh" (Worker.executable worker) (Worker.script worker) Nothing R.Serial (base </> "learner.sh") Learner.Process (Worker.cache worker) (R.sessions sessions)
        config = Loop.Config backend (base </> "run") (base </> "input") (base </> "reference") settings Store.RenameExclusive
    evalIO (writeFile (base </> "learner.sh") (copying ++ script))
    pure (Fixture base (Runtime.Run config 0 [const (Right cycle')] [members]) (Worker.script worker))

ran :: PropertyT IO Fixture
ran = do
    fixture <- prepared
    outcome <- evalIO (Runtime.run (chosen fixture) [])
    either (\problem -> annotateShow problem >> failure) pure outcome
    pure fixture

located :: Runtime.Run -> FilePath -> Runtime.Run
located (Runtime.Run config lag workload sizes) directory = Runtime.Run config {Loop.root = directory} lag workload sizes

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
resumed fixture directory = Runtime.resume directory (const (Right (located (chosen fixture) directory)))

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
    outcome <- evalIO (Runtime.run (chosen fixture) [])
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
