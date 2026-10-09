{-# LANGUAGE OverloadedStrings #-}

module Journals (journals) where

import Control.Exception (ErrorCall (..), try)
import Control.Monad (forM_)
import Data.Aeson (Value (..), object, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as Char
import Data.Either (isLeft)
import Hedgehog
import Invar.Async.Entry qualified as Entry
import Invar.History.Runtime qualified as History
import Invar.Journal qualified as Journal
import Invar.Runtime qualified as Runtime
import Invar.Transcript qualified as Transcript
import Invar.Workload qualified as Workload
import Sessions qualified
import Store (workspace)
import System.Directory (canonicalizePath, createDirectory, doesFileExist, getCurrentDirectory)
import System.FilePath ((</>))
import System.IO.Error (ioeGetErrorString, isAlreadyExistsError, tryIOError)

journals :: Group
journals =
    Group
        "Durable run journal"
        [ ("a journal starts with its declaration, is created once, keeps appended entries in order and refuses entries once closed", withTests 1 (property ordered))
        , ("a resumed journal drops an unterminated final line only after its entries are admitted, leaves a refused journal unchanged and needs a complete declaration", withTests 1 (property resumed))
        , ("resuming a run keeps the caller's working directory when it refuses or fails", withTests 1 (property directories))
        , ("resuming or inspecting a directory without a journal is refused because the journal is missing, and creates none", withTests 1 (property missing))
        , ("only an unterminated final line is dropped; any other malformed line is refused", withTests 1 (property damaged))
        , ("a transcript is created once, keeps its lines and the bytes after the last newline exactly and journals how its process ended; a closed journal closes the transcripts left open and opens no more", withTests 1 (property transcribed))
        ]

entry :: Int -> Value
entry index = object ["entry" .= ("event" :: String), "index" .= index]

ordered :: PropertyT IO ()
ordered = do
    root <- workspace
    let path = root </> "journal.jsonl"
        declaration = object ["entry" .= ("declaration" :: String), "staleness" .= (1 :: Int)]
    (again, closed) <- evalIO $ do
        kept <- Journal.with path declaration $ \journal -> do
            mapM_ (Journal.append journal . entry) [0 .. 4]
            again <- tryIOError (Journal.with path declaration (const (pure ())))
            pure (journal, again)
        closed <- tryIOError (Journal.append (fst kept) (entry 5))
        pure (snd kept, closed)
    case again of
        Left problem -> assert (isAlreadyExistsError problem)
        Right () -> annotate "a second journal was created over the first" >> failure
    assert (isLeft closed)
    encoded <- evalIO (Bytes.readFile path)
    decoded <- evalEither (Journal.entries encoded)
    map Object decoded === declaration : map entry [0 .. 4]

resumed :: PropertyT IO ()
resumed = do
    root <- workspace
    let path = root </> "journal.jsonl"
        declaration = object ["entry" .= ("declaration" :: String)]
    evalIO (Journal.with path declaration (\journal -> Journal.append journal (entry 0)))
    evalIO (Bytes.appendFile path "{\"entry\":\"ev")
    unfinished <- evalIO (Bytes.readFile path)
    refused <- evalIO (Journal.resume path (const (pure (Left "refused"))) (\_ _ -> pure ()))
    refused === Left ("refused" :: String)
    evalIO (Bytes.readFile path) >>= (=== unfinished)
    seen <- evalIO (Journal.resume path (pure . Right) (\recorded journal -> Journal.append journal (entry 1) >> pure recorded))
    fmap (map Object) seen === (Right [declaration, entry 0] :: Either String [Value])
    after <- evalIO (Bytes.readFile path) >>= evalEither . Journal.entries
    map Object after === [declaration, entry 0, entry 1]
    forM_ ["", "{\"entry\":\"decl"] $ \contents -> do
        let partial = root </> "partial.jsonl"
        evalIO (Bytes.writeFile partial contents)
        incomplete <- evalIO (tryIOError (Journal.resume partial (pure . Right) (\_ _ -> pure ())))
        assert (isLeft (incomplete :: Either IOError (Either String ())))

directories :: PropertyT IO ()
directories = do
    root <- workspace
    started <- workspace
    let output = root </> "run"
    evalIO (createDirectory output)
    document <- evalEither Sessions.workload
    evalIO (Journal.with (output </> "journal.jsonl") (Entry.encode (Entry.Declared started (Fields.fromList ["arguments" .= Null, "workload" .= Workload.value document]))) (const (pure ())))
    caller <- evalIO getCurrentDirectory
    refused <- evalIO (Runtime.resume output (const (Left "refused")))
    case refused of
        Left (Runtime.Declaration "refused") -> success
        _ -> annotate "the declaration was not refused" >> failure
    evalIO getCurrentDirectory >>= (=== caller)
    failed <- evalIO (try (Runtime.resume output (const (error "interpretation failed"))))
    case failed of
        Left (ErrorCall _) -> success
        Right _ -> annotate "the failing interpretation returned" >> failure
    evalIO getCurrentDirectory >>= (=== caller)

missing :: PropertyT IO ()
missing = do
    root <- workspace
    let output = root </> "run"
        journal = output </> "journal.jsonl"
    evalIO (createDirectory output)
    resuming <- evalIO (tryIOError (Runtime.resume output (const (Left "unread"))))
    inspecting <- evalIO (tryIOError (History.inspect output (const (Left "unread"))))
    target <- evalIO (canonicalizePath output)
    let named = Left ("There is no run journal at " ++ target </> "journal.jsonl")
    either (Left . ioeGetErrorString) (const (Right ())) resuming === named
    either (Left . ioeGetErrorString) (const (Right ())) inspecting === named
    evalIO (doesFileExist journal) >>= (=== False)

damaged :: PropertyT IO ()
damaged = do
    let line value = Char.pack (show value)
        complete = Char.unlines ["{\"entry\":\"declaration\"}", "{\"entry\":\"event\",\"index\":0}"]
    kept <- evalEither (Journal.entries (complete <> "{\"entry\":\"ev"))
    length kept === 2
    fmap (map (Fields.lookup "entry")) (Journal.entries complete) === Right [Just (String "declaration"), Just (String "event")]
    Journal.entries "" === Right []
    assert (isLeft (Journal.entries (Char.unlines ["{\"entry\":\"declaration\"}", "", "{\"entry\":\"event\"}"])))
    assert (isLeft (Journal.entries (Char.unlines ["{\"entry\":\"declaration\"}", "{\"entry\":", "{\"entry\":\"event\"}"])))
    assert (isLeft (Journal.entries (Char.unlines ["{\"entry\":\"declaration\"}", line (3 :: Int)])))

transcribed :: PropertyT IO ()
transcribed = do
    root <- workspace
    let path = root </> "journal.jsonl"
        declaration = object ["entry" .= ("declaration" :: String)]
        first' = root </> "0.jsonl"
        ending = Entry.encode . Entry.Finished 0
    (again, late, (journal', left)) <- evalIO $ Journal.with path declaration $ \journal -> do
        transcript <- Journal.transcript journal first' ending
        Transcript.record transcript "{\"line\":0}"
        Journal.append journal (entry 0)
        Transcript.record transcript "{\"line\":1}"
        Transcript.partial transcript "{\"li"
        Transcript.finished transcript (Transcript.Stopped Transcript.Cut)
        late <- tryIOError (Transcript.record transcript "{\"line\":2}")
        again <- tryIOError (Journal.transcript journal first' ending)
        left <- Journal.transcript journal (root </> "1.jsonl") ending
        pure (again, late, (journal, left))
    case again of
        Left problem -> assert (isAlreadyExistsError problem)
        Right _ -> annotate "a transcript was created over an earlier one" >> failure
    assert (isLeft late)
    closed <- evalIO (tryIOError (Transcript.record left "{}"))
    assert (isLeft closed)
    refused <- evalIO (tryIOError (Journal.transcript journal' (root </> "2.jsonl") ending))
    assert (isLeft refused)
    evalIO (Bytes.readFile first') >>= (=== "{\"line\":0}\n{\"line\":1}\n{\"li")
    recorded <- evalIO (Bytes.readFile path) >>= evalEither . Journal.entries
    map Object recorded === [declaration, entry 0, ending (Transcript.Stopped Transcript.Cut)]
