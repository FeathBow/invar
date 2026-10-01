{-# LANGUAGE OverloadedStrings #-}

module Journals (journals) where

import Data.Aeson (Value (..), object, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as Char
import Data.Either (isLeft)
import Hedgehog
import Invar.Journal qualified as Journal
import Store (workspace)
import System.FilePath ((</>))
import System.IO.Error (isAlreadyExistsError, tryIOError)

journals :: Group
journals =
    Group
        "Durable run journal"
        [ ("a journal starts with its declaration, is created once, keeps appended entries in order and refuses entries once closed", withTests 1 (property ordered))
        , ("a stored file holds exactly its contents and is never replaced", withTests 1 (property stored))
        , ("only an unterminated final line is dropped; any other malformed line is refused", withTests 1 (property damaged))
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
