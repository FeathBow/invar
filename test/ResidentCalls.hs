{-# LANGUAGE OverloadedStrings #-}

module ResidentCalls (residentCalls) where

import Calls qualified as Fixture
import Control.Monad (forM_, (>=>))
import Data.Aeson (Value (..), object, toJSON, withObject, (.:), (.=))
import Data.Aeson.Key qualified as Key
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString.Char8 qualified as Bytes
import Data.Maybe (listToMaybe)
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8)
import Hedgehog
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Spec.Invocation qualified as Invocation
import Invar.Spec.Load qualified as Load
import Invar.Transcript qualified as Transcript
import Invar.Worker qualified as Worker
import ResidentFixture qualified as F
import Store (workspace)
import System.Directory (doesFileExist)
import System.FilePath ((</>))

residentCalls :: Group
residentCalls =
    Group
        "Acknowledged resident inference"
        [ ("successive groups retain distinct receipts after exact activation release", once completed)
        , ("acknowledgement requires exact owner digest load inventory and measurement", once release)
        , ("retired activation identities remain unavailable for replay", once replay)
        , ("initial load and subsequent activation observations remain distinct", once loading)
        , ("resident owner completion requires exact final close and child exit, and output after the close is recorded", once closing)
        , ("a transcript write that fails ends the transcript: nothing more is written, the output is cut and the original error surfaces", once unwritten)
        ]
  where
    once = withTests 1 . property

completed :: PropertyT IO ()
completed = do
    root <- workspace
    first <- F.prepare root 0 [1, 0]
    second <- F.prepare root 1 [3, 2]
    (returned, emitted) <- F.run root (F.scenario [first, second])
    receipts <- evalEither returned
    let expected = concatMap F.calls [first, second]
    map Trajectory.binding receipts === map Call.binding expected
    map (Invocation.completedBinding . Load.report . Trajectory.loaded) receipts === map Call.binding expected
    map Trajectory.behaviorBits receipts === replicate 4 [0xbf000000, 0xbe800000]
    evalIO (length . lines <$> readFile (root </> "pids")) >>= (=== 1)
    assert (Fixture.wire (F.before second) `Bytes.isInfixOf` emitted)

release :: PropertyT IO ()
release = do
    root <- workspace
    original <- F.prepare root 0 [1, 0]
    loads <- evalEither (parseEither (withObject "fixture acknowledgement" (.: "loads")) (F.released original))
    let changed key value = Fixture.change key value (F.released original)
        inventories = [[], take 1 loads, reverse loads, loads ++ take 1 loads, map (Fixture.change "program" (String "wrong")) loads]
        invalid = [changed "owner" (object ["role" .= ("learning" :: Text), "session" .= F.owner]), changed "result_sha256" (String (Text.replicate 64 "0")), changed "stage" (String "closed"), changed "measurement" (String "{}\n"), changed "measurement" (String (decodeUtf8 (Fixture.wire [Fixture.change "cpu_seconds" (Number (-1)) (F.timer "released")]))), without "measurement" (F.released original)] ++ map (changed "loads" . toJSON) inventories
    forM_ invalid $ \ack -> F.run root (F.scenario [original {F.released = ack}]) >>= rejected . fst

replay :: PropertyT IO ()
replay = do
    root <- workspace
    first <- F.prepare root 0 [1, 0]
    second <- F.prepare root 1 [1, 0]
    (returned, _) <- F.run root (F.scenario [first, second])
    rejected returned
    evalIO (doesFileExist (root </> "approved0")) >>= (=== True)
    evalIO (doesFileExist (root </> "approved1")) >>= (=== False)

loading :: PropertyT IO ()
loading = do
    root <- workspace
    first <- F.prepare root 0 [1, 0]
    second <- F.prepare root 1 [3, 2]
    let wrongFirst = first {F.before = F.timer "activation" : drop 1 (F.before first)}
        wrongSecond = second {F.before = F.timer "load" : drop 1 (F.before second)}
    forM_ [[wrongFirst], [first, wrongSecond], [first, second {F.before = drop 1 (F.before second)}]] $ \groups -> F.run root (F.scenario groups) >>= rejected . fst

closing :: PropertyT IO ()
closing = do
    root <- workspace
    group <- F.prepare root 0 [0, 1]
    let original = F.scenario [group]
        invalid = [original {F.closed = Fixture.change "groups" (Number 0) (F.closed original)}, original {F.closed = without "measurement" (F.closed original)}, original {F.ending = "exit 7"}]
    forM_ invalid (F.run root >=> rejected . fst)
    (outcome, emitted) <- F.run root original {F.ending = "printf '%s\\n' trailing\nexit 0"}
    rejected outcome
    last (Bytes.lines emitted) === "trailing"

without :: Text -> Value -> Value
without key (Object fields) = Object (Fields.delete (Key.fromText key) fields)
without _ value = value

rejected :: Either Worker.Failure value -> PropertyT IO ()
rejected (Left _) = success
rejected (Right _) = failure

unwritten :: PropertyT IO ()
unwritten = do
    root <- workspace
    group <- F.prepare root 0 [0, 1]
    refused <- evalMaybe (listToMaybe (Bytes.lines (Fixture.wire (F.after group))))
    (thrown, emitted, ending) <- F.interrupted root (F.scenario [group]) refused
    thrown === Just "The transcript refused a write"
    emitted === Fixture.wire (F.before group) <> Bytes.take 1 refused
    ending === Just (Transcript.Stopped Transcript.Cut)
