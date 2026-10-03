{-# LANGUAGE OverloadedStrings #-}

module Journals (journals) where

import Control.Exception (ErrorCall (..), try)
import Control.Monad (forM_)
import Data.Aeson (Value (..), encode, object, toJSON, (.=))
import Data.Aeson.Key qualified as Key
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as Char
import Data.ByteString.Lazy qualified as Lazy
import Data.Either (isLeft)
import Hedgehog hiding (Command, Update)
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Async.Core (Attempt (..), Command (..))
import Invar.Async.Plan (Request (..), Update (..), Version (..))
import Invar.Async.Record qualified as Record
import Invar.Journal qualified as Journal
import Invar.Runtime qualified as Runtime
import Invar.Spec.Invocation qualified as V
import Invar.Transcript qualified as Transcript
import Numeric.Natural (Natural)
import Store (workspace)
import System.Directory (createDirectory, getCurrentDirectory)
import System.Exit (ExitCode (..))
import System.FilePath ((</>))
import System.IO.Error (isAlreadyExistsError, tryIOError)

journals :: Group
journals =
    Group
        "Durable run journal"
        [ ("a journal starts with its declaration, is created once, keeps appended entries in order and refuses entries once closed", withTests 1 (property ordered))
        , ("a stored file holds exactly its contents and is never replaced", withTests 1 (property stored))
        , ("a resumed journal drops an unterminated final line before appending and needs a complete declaration", withTests 1 (property resumed))
        , ("resuming a run keeps the caller's working directory when it refuses or fails", withTests 1 (property directories))
        , ("only an unterminated final line is dropped; any other malformed line is refused", withTests 1 (property damaged))
        , ("every journal entry reads back as the entry that was written", withTests 300 (property typed))
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

stored :: PropertyT IO ()
stored = do
    root <- workspace
    let path = root </> "request0.json"
        contents = Bytes.concat (replicate 4096 "{\"result\":true}")
    evalIO (Journal.store path contents)
    again <- evalIO (tryIOError (Journal.store path "{}"))
    case again of
        Left problem -> assert (isAlreadyExistsError problem)
        Right () -> annotate "a stored file was replaced" >> failure
    evalIO (Bytes.readFile path) >>= (=== contents)

resumed :: PropertyT IO ()
resumed = do
    root <- workspace
    let path = root </> "journal.jsonl"
        declaration = object ["entry" .= ("declaration" :: String)]
    evalIO (Journal.with path declaration (\journal -> Journal.append journal (entry 0)))
    evalIO (Bytes.appendFile path "{\"entry\":\"ev")
    seen <- evalIO (Journal.resume path (\recorded journal -> Journal.append journal (entry 1) >> pure recorded))
    map Object seen === [declaration, entry 0]
    after <- evalIO (Bytes.readFile path) >>= evalEither . Journal.entries
    map Object after === [declaration, entry 0, entry 1]
    forM_ ["", "{\"entry\":\"decl"] $ \contents -> do
        let partial = root </> "partial.jsonl"
        evalIO (Bytes.writeFile partial contents)
        refused <- evalIO (tryIOError (Journal.resume partial (\_ _ -> pure ())))
        assert (isLeft refused)

directories :: PropertyT IO ()
directories = do
    root <- workspace
    started <- workspace
    let output = root </> "run"
    evalIO (createDirectory output)
    evalIO (Journal.with (output </> "journal.jsonl") (object ["entry" .= ("declaration" :: String), "directory" .= started]) (const (pure ())))
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

typed :: PropertyT IO ()
typed = do
    written <- forAll entries
    decoded <- evalEither (Journal.entries (Lazy.toStrict (encode (Record.encode written)) <> "\n"))
    traverse (parseEither Record.decode) decoded === Right [written]
  where
    number = Gen.integral (Range.linear 0 1000) :: Gen Natural
    text = Gen.string (Range.linear 0 12) Gen.alphaNum
    bound = V.Binding . V.CallId <$> number <*> (V.AttemptId <$> number) <*> (V.Instance <$> number)
    command = Gen.choice [Dispatch . Request <$> number <*> (Version <$> number), Send <$> update <*> attempt, Open <$> update <*> attempt <*> number <*> text, Record <$> update <*> attempt <*> text, Commit <$> update <*> attempt]
    update = Update <$> number
    attempt = Attempt <$> number
    claim =
        Gen.choice
            [ Record.Connected <$> number <*> number
            , Record.Lost <$> number <*> number
            , Record.Started <$> number <*> number <*> number
            , Record.Completed <$> number <*> number <*> number <*> text
            , Record.Ready <$> number <*> number <*> bound <*> text <*> text
            , Record.Current <$> number <*> number <*> number <*> text
            , Record.Applied <$> number <*> number <*> bound <*> text <*> number <*> text <*> text <*> text
            , Record.Staged <$> number <*> number <*> text
            , Record.Recorded <$> number <*> number
            , Record.Committed <$> number <*> number
            , Record.Abandoned <$> number <*> number
            ]
    read' = Gen.element [Transcript.Complete, Transcript.Cut]
    ended = Gen.choice [Transcript.Unlaunched <$> text, Transcript.Exited ExitSuccess <$> read', Transcript.Exited . ExitFailure <$> Gen.filter (/= 0) (Gen.int (Range.linear (-64) 255)) <*> read', Transcript.Stopped <$> read']
    declared = Fields.fromList <$> Gen.list (Range.linear 0 4) ((,) . Key.fromString <$> Gen.filter (`notElem` ["entry", "directory"]) text <*> (toJSON <$> text))
    entries =
        Gen.choice
            [ Record.Declared <$> text <*> declared
            , Record.Opened <$> number <*> number
            , Record.Dispatched <$> number <*> number <*> number <*> bound
            , Record.Attempted <$> number <*> number <*> bound
            , Record.Reserved <$> number <*> Gen.element [Record.Inference, Record.Learner] <*> number <*> number
            , Record.Finished <$> number <*> ended
            , Record.Happened <$> claim <*> Gen.list (Range.linear 0 4) command
            , Record.Stored <$> number <*> bound <*> text
            , Record.Verified <$> number <*> number <*> text
            , Record.Resumed <$> Gen.list (Range.linear 0 4) number <*> number <*> number <*> number
            , Record.Elapsed <$> Gen.element ["rollout", "learner"] <*> number <*> Gen.double (Range.linearFrac 0 1.0e6) <*> Gen.double (Range.linearFrac 0 1.0e6)
            ]

transcribed :: PropertyT IO ()
transcribed = do
    root <- workspace
    let path = root </> "journal.jsonl"
        declaration = object ["entry" .= ("declaration" :: String)]
        first' = root </> "0.jsonl"
        ending = Record.encode . Record.Finished 0
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
