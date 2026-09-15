{-# LANGUAGE OverloadedStrings #-}

module ResidentReplays (residentReplays, streams) where

import BatchedObservations qualified as Batch
import Calls (change, field, request, wire)
import Control.Monad (foldM, forM_)
import Data.Aeson (Value (..), eitherDecodeStrict, object, parseJSON, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8, encodeUtf8)
import Data.Word (Word32)
import GHC.Float (castWord32ToFloat, float2Double)
import Hedgehog
import Invar.Evaluation qualified as Evaluation
import Invar.Infer qualified as Infer
import Invar.Replay.Inference qualified as Replay
import Numeric.Natural (Natural)
import ResidentObservations qualified as Fixture
import Updates (alter)
import Workloads (array)

residentReplays :: Group
residentReplays =
    Group
        "Complete resident direct replay observations"
        [ ("direct owners replay the original cohort partition and complete lifetime", once complete)
        , ("each replay needs the explicit owner complete result release close and exit", once boundaries)
        , ("a valid changed numerical result is reported without claiming equality", once numerical)
        , ("resident replay requires a physical reference and admits unused owners explicitly", once declarations)
        ]
  where
    once = withTests 1 . property

run :: Evaluation.Run
run = Evaluation.Run (Infer.artifact request) 0

streams :: [Value] -> Either String [(Natural, ByteString)]
streams events = do
    (_, accepted) <- foldM accumulate ([], Map.empty) events
    pure (Map.toAscList (fmap wire accepted))
  where
    accumulate (pending, accepted) value
        | stage "released" value = do
            owner <- parseEither parseJSON (field "session" (field "owner" value))
            pure ([], Map.insertWith (flip (++)) owner (reverse pending ++ [value]) accepted)
        | stage "closed" value = do
            owner <- parseEither parseJSON (field "session" (field "owner" value))
            pure ([], Map.insertWith (flip (++)) owner [value] accepted)
        | phase value = pure ([], accepted)
        | otherwise = pure (value : pending, accepted)

complete :: PropertyT IO ()
complete = forM_ [1, 2, 3] $ \count -> do
    (tasks, events) <- Fixture.fixture count
    reference <- evalEither (Replay.admit tasks (run, Replay.Resident) (wire events))
    owners <- evalMaybe (Replay.ownerCalls reference)
    outputs <- evalEither (streams events)
    length owners === count
    field "sessions" (field "residence" (Replay.describe reference)) === toJSON count
    forM_ (zip owners outputs) $ \((owner, calls), (actualOwner, raw)) -> do
        owner === actualOwner
        observed <- evalEither (Replay.observeResident (owner, 0) calls raw)
        field "equal_results" observed === toJSON (length calls)
        field "response_tokens" observed === toJSON (length calls * 2)
        field "close" observed === object ["cpu_seconds" .= (0.125 :: Double)]
        let measured = array (field "groups" observed)
        length measured === if null calls then 0 else 3
        forM_ (zip [0 :: Int ..] measured) $ \(index, group) -> do
            field "cohort" group === toJSON index
            field "load" (field "observation" group) === if index == 0 then object ["cpu_seconds" .= Number 1] else Null

selected :: PropertyT IO ([Replay.Call], [Value])
selected = do
    (tasks, events) <- Fixture.fixture 1
    reference <- evalEither (Replay.admit tasks (run, Replay.Resident) (wire events))
    owners <- evalMaybe (Replay.ownerCalls reference)
    outputs <- evalEither (streams events)
    case (owners, outputs) of
        ([(0, calls)], [(0, raw)]) -> (calls,) <$> traverse (evalEither . eitherDecodeStrict) (Bytes.lines raw)
        _ -> failure

boundaries :: PropertyT IO ()
boundaries = do
    (calls, events) <- selected
    let raw = wire events
    rejected (Replay.observeResident (9, 0) calls raw)
    rejected (Replay.observeResident (0, 7) calls raw)
    rejected (Replay.observe (Replay.Resident, 0) calls raw)
    rejected (Replay.observeResident (0, 0) (reverse (take 2 calls) ++ drop 2 calls) raw)
    rejected (Replay.observeResident (0, 0) calls (Bytes.init raw))
    forM_ ["load", "activation", "released", "closed"] $ \name -> do
        index <- evalMaybe (firstIndex (stage name) events)
        rejected (Replay.observeResident (0, 0) calls (wire (take index events ++ drop (index + 1) events)))
        rejected (Replay.observeResident (0, 0) calls (wire (take index events ++ [events !! index] ++ drop index events)))

numerical :: PropertyT IO ()
numerical = do
    (calls, events) <- selected
    index <- evalMaybe (firstIndex (stage "result") events)
    changed <- Fixture.reseal (alter index (\frame -> change "calls" (toJSON (alter 0 changedMember (array (field "calls" frame)))) frame) events)
    observed <- evalEither (Replay.observeResident (0, 0) calls (wire changed))
    field "equal_results" observed === toJSON (length calls - 1)
    field "response_tokens" observed === toJSON (length calls * 2)

changedMember :: Value -> Value
changedMember (String raw) = case eitherDecodeStrict (encodeUtf8 raw) of
    Right result ->
        let behaviorWords = toJSON changedWord : drop 1 (array (field "behavior_bits" result))
            values = toJSON (float2Double (castWord32ToFloat changedWord)) : drop 1 (array (field "behavior" result))
            changed = change "behavior_bits" (toJSON behaviorWords) (change "behavior" (toJSON values) result)
         in String (decodeUtf8 (wire [changed]))
    Left problem -> error problem
  where
    changedWord :: Word32
    changedWord = 3204448257
changedMember _ = error "Expected original result JSONL"

declarations :: PropertyT IO ()
declarations = do
    (tasks, events, _) <- Batch.fixture
    rejected (Replay.admit tasks (run, Replay.Resident) (wire events))
    rejected (Replay.decodeCalls "[]")
    unused <- evalEither (Replay.decodeResidentCalls "[]")
    (residentTasks, residentEvents) <- Fixture.fixture 3
    _ <- evalEither (Replay.admit residentTasks (run, Replay.Resident) (wire residentEvents))
    outputs <- evalEither (streams residentEvents)
    raw <- evalMaybe (lookup 2 outputs)
    observed <- evalEither (Replay.observeResident (2, 0) unused raw)
    field "calls" observed === toJSON ([] :: [Value])
    rejected (Replay.observeResident (2, 0) unused (wire [object ["stage" .= String "load", "cpu_seconds" .= Number 1]] <> raw))

stage :: Text -> Value -> Bool
stage expected (Object fields) = Fields.lookup "stage" fields == Just (String expected)
stage _ _ = False

phase :: Value -> Bool
phase (Object fields) = Fields.member "phase" fields
phase _ = False

firstIndex :: (value -> Bool) -> [value] -> Maybe Int
firstIndex predicate values = case [index | (index, value) <- zip [0 ..] values, predicate value] of
    index : _ -> Just index
    [] -> Nothing

rejected :: Either String value -> PropertyT IO ()
rejected (Left _) = success
rejected (Right _) = failure
