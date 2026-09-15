{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

module BatchedObservations (batchedObservations, fixture) where

import Calls (change, field, request, wire)
import Control.Monad (forM_)
import Data.Aeson (Value (..), object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8, encodeUtf8)
import Hedgehog
import Invar.Evaluation qualified as Evaluation
import Invar.Infer qualified as Infer
import Invar.Measurement.Inference qualified as Measurement
import Invar.Replay.Inference qualified as Replay
import Invar.Workload qualified as Workload
import Measurements qualified
import Updates (alter)
import Workloads (array, replace)

batchedObservations :: Group
batchedObservations =
    Group
        "Finite batched observation readers"
        [ ("shared inference duration is counted once for all batch members", once measured)
        , ("batch replay retains every call and requires its explicit execution mode", once replayed)
        , ("missing extra repeated or reordered nested members cannot be admitted", once inventory)
        , ("group boundaries and shared measurement inventory remain strict", once boundaries)
        , ("batch observations preserve original probability zero signs", once signedZero)
        ]
  where
    once = withTests 1 . property

frame :: Text -> [ByteString] -> Value
frame stage members = object ["stage" .= stage, "format" .= String "invar-inference-batch-v1", "calls" .= map decodeUtf8 members]

fixture :: PropertyT IO (Workload.Document, [Value], [Value])
fixture = do
    (tasks, serial) <- Measurements.fixture
    case serial of
        [profile, load, firstLoaded, firstConsumed, timed, firstResult, _, secondLoaded, secondConsumed, _, secondResult, summary, completed] -> do
            let ready = frame "consumed" (map wire [[firstLoaded, firstConsumed], [secondLoaded, secondConsumed]])
                result = frame "result" (map (wire . pure) [firstResult, secondResult])
                grouped = [object ["stage" .= String "loading"], profile, load, ready, timed, result, summary, completed]
            pure (tasks, grouped, serial)
        _ -> failure

run :: Evaluation.Run
run = Evaluation.Run (Infer.artifact request) 0

measured :: PropertyT IO ()
measured = do
    (tasks, events, serial) <- fixture
    original <- evalEither (Measurement.admit tasks run (wire serial))
    grouped <- evalEither (Measurement.admit tasks run (wire events))
    field "critical_path_seconds" (Measurement.describe original) === Number 5
    let reported = Measurement.describe grouped
    field "critical_path_seconds" reported === Number 3
    field "sessions_per_cohort" reported === toJSON [1 :: Int]
    case (array (field "measurements" reported), array (field "loads" reported)) of
        ([batch], [load]) -> do
            field "inference" batch === object ["cpu_seconds" .= Number 2]
            length (array (field "calls" batch)) === 2
            forM_ (array (field "calls" batch)) $ \case
                Object fields -> assert (not (Fields.member "inference" fields))
                _ -> failure
            field "inference_seconds" load === Number 2
            field "requests_per_execution" load === toJSON [2 :: Int]
            field "calls" load === Number 2
        _ -> failure

replayed :: PropertyT IO ()
replayed = do
    (tasks, events, serial) <- fixture
    reference <- evalEither (Replay.admit tasks (run, Replay.Batched) (wire events))
    let calls = Replay.calls reference
    length calls === 2
    actual <- evalEither (Replay.observe (Replay.Batched, 0) calls (wire (take 6 events)))
    field "load" actual === object ["cpu_seconds" .= Number 1]
    field "inference" actual === object ["cpu_seconds" .= Number 2]
    forM_ (array (field "calls" actual)) $ \member -> field "result_equal" member === Bool True
    rejected (Replay.observe (Replay.Session, 0) calls (wire (take 6 events)))
    rejected (Replay.observe (Replay.Batched, 0) calls (wire (take 11 serial)))
    rejected (Replay.observe (Replay.Batched, 7) calls (wire (take 6 events)))
    rejected (Replay.observe (Replay.Batched, 0) (reverse calls) (wire (take 6 events)))

inventory :: PropertyT IO ()
inventory = do
    (tasks, events, _) <- fixture
    forM_ [3, 5] $ \position -> do
        let members = array (field "calls" (events !! position))
            invalid = [[], take 1 members, members ++ take 1 members, reverse members, take 1 members ++ take 1 members]
        forM_ invalid $ \changed -> reject tasks (alter position (change "calls" (toJSON changed)) events)
        reject tasks (alter position (change "extra" Null) events)
        reject tasks (alter position (change "format" (String "other")) events)
    let wrongModel = alter 7 (change "assembly" (String "wrong")) events
    reject tasks wrongModel

boundaries :: PropertyT IO ()
boundaries = do
    (tasks, events, _) <- fixture
    forM_ [3, 4, 5] $ \position -> do
        reject tasks (take position events ++ drop (position + 1) events)
        reject tasks (take position events ++ [events !! position] ++ drop position events)
    reject tasks (alter 4 (change "cpu_seconds" (Number (-1))) events)
    reject tasks (alter 4 (change "seconds" (Number 1)) events)
    reject tasks (alter 7 (change "sessions" (Number 2)) events)
    reject tasks (take 6 events ++ take 6 events ++ drop 6 events)
    reject tasks (alter 4 (change "phase" (String "cycle")) events)

signedZero :: PropertyT IO ()
signedZero = do
    (tasks, events, _) <- fixture
    let results = array (field "calls" (events !! 5))
        rewrite (String raw) = String (decodeUtf8 (replace "[-0.5,-0.25]" "[-0.0,-0.25]" (replace "[3204448256,3196059648]" "[2147483648,3196059648]" (encodeUtf8 raw))))
        rewrite _ = error "Expected original encoded result text"
        changed = alter 5 (change "calls" (toJSON (map rewrite results))) events
    _ <- evalEither (Measurement.admit tasks run (wire changed))
    let wrong (String raw) = String (decodeUtf8 (replace "[-0.0,-0.25]" "[0.0,-0.25]" (encodeUtf8 raw)))
        wrong _ = error "Expected original encoded result text"
    reject tasks (alter 5 (change "calls" (toJSON (map (wrong . rewrite) results))) events)

reject :: Workload.Document -> [Value] -> PropertyT IO ()
reject tasks events = rejected (Measurement.admit tasks run (wire events))

rejected :: Either String value -> PropertyT IO ()
rejected (Left _) = success
rejected (Right _) = failure
