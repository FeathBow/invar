{-# LANGUAGE OverloadedStrings #-}

module InferenceObservations (inferenceObservations) where

import Calls (change, field, request, setup, wire)
import Control.Monad (forM_)
import Data.Aeson (Value (..), object, toJSON, (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text qualified as Text
import Data.Word (Word32)
import Hedgehog
import Invar.Infer qualified as Infer
import Invar.Infer.Observation qualified as Observation
import Invar.Infer.Result qualified as Result
import Invar.Spec.Invocation qualified as V
import Workloads (replace)

inferenceObservations :: Group
inferenceObservations = Group "Complete inference observations" [("complete observations retain bound results and exact source identity", once matching), ("repeated missing reordered and trailing records fail", once complete), ("loads programs bindings and model images must correspond", once correspondence), ("strict result field and raw JSON schemas apply", once malformed), ("behavior zero signs retain actual floating literals", once signedZero)]
  where
    once = withTests 1 . property

bound :: V.Binding
bound = V.Binding (V.CallId 7) (V.AttemptId 11) (V.Instance 13)

fixture :: PropertyT IO [Value]
fixture = do
    (_, events) <- setup
    case events of
        [loaded, consumed, output] -> pure [change "model" (String "test-model") (change "revision" (String "test-revision") loaded), consumed, object ["stage" .= String "inference", "cpu_seconds" .= Number 1], output]
        _ -> failure

observe :: ByteString -> Either String Observation.Report
observe encoded = do
    planned <- either (Left . show) Right (Infer.prepare request)
    Observation.admit planned bound encoded

matching :: PropertyT IO ()
matching = do
    events <- fixture
    original <- evalEither (observe (wire events))
    Result.response (Observation.result original) === "#### 12"
    Observation.binding original === bound
    changed <- evalEither (observe (wire (object ["stage" .= String "load"] : events)))
    assert (Observation.logDigest original /= Observation.logDigest changed)
    Observation.result original === Observation.result changed

complete :: PropertyT IO ()
complete = do
    events <- fixture
    forM_ [[], take 3 events, drop 1 events, reverse events, events ++ events, events ++ [object ["stage" .= String "load"]]] (reject . wire)
    forM_ [0 .. length events - 1] $ \index -> reject (wire (take index events ++ drop (index + 1) events))
    reject (Bytes.init (wire events))

correspondence :: PropertyT IO ()
correspondence = do
    events <- fixture
    forM_ [0, 1, 3] $ \index ->
        forM_ ["call", "attempt", "instance"] $ \axis ->
            reject (wire (alter index (\value -> change "binding" (change axis (Number 99) (field "binding" value)) value) events))
    forM_ [0, 1] $ \index ->
        reject (wire (alter index (\value -> change "load" (change "program" (String "wrong") (field "load" value)) value) events))
    reject (wire (alter 1 (change "program" (String "wrong")) events))
    forM_ ["artifact", "profile"] $ \axis ->
        reject (wire (alter 0 (\value -> change "image" (change axis (String "wrong") (field "image" value)) value) events))
    forM_ ["model", "revision"] $ \axis -> reject (wire (alter 0 (change axis (String "")) events))
    forM_ ["tokenizer", "base", "assembly"] $ \axis ->
        forM_ [0, 1, 3] $ \index -> reject (wire (alter index (change axis (String (Text.replicate 64 "b"))) events))

malformed :: PropertyT IO ()
malformed = do
    events <- fixture
    reject (replace "\"call\":7" "\"call\":7,\"call\":7" (wire events))
    forM_ [Null, Bool True, Number 0.5, Number (-1)] $ \value -> reject (wire (alter 3 (change "prompt_length" value) events))
    forM_ ["tokens", "behavior", "behavior_bits", "request", "binding"] $ \key ->
        reject (wire (alter 3 (omit key) events))
    reject (wire (alter 3 (change "extra" Null) events))
    reject (wire (alter 1 (change "extra" Null) events))

signedZero :: PropertyT IO ()
signedZero = do
    events <- fixture
    let negativeWord = 0x80000000 :: Word32
        quarterWord = 0xbe800000 :: Word32
        withBits word = wire (alter 3 (change "behavior_bits" (toJSON [word, quarterWord])) events)
        spelling text word = replace "[-0.5,-0.25]" text (withBits word)
    forM_ [("[-0.0,-0.25]", negativeWord), ("[0.0,-0.25]", 0), ("[0,-0.25]", 0), ("[-0,-0.25]", 0)] $ \(text, word) -> do
        actual <- evalEither (observe (spelling text word))
        Result.behaviorBits (Observation.result actual) === [word, quarterWord]
    forM_ [("[0.0,-0.25]", negativeWord), ("[-0.0,-0.25]", 0), ("[-0,-0.25]", negativeWord)] $ \(text, word) -> reject (spelling text word)

alter :: Int -> (value -> value) -> [value] -> [value]
alter selected changeValue = zipWith (\index value -> if index == selected then changeValue value else value) [0 ..]

omit :: Key -> Value -> Value
omit key (Object fields) = Object (Fields.delete key fields)
omit _ value = value

reject :: ByteString -> PropertyT IO ()
reject encoded = case observe encoded of
    Left _ -> success
    Right report -> annotateShow (Observation.describe report) >> failure
