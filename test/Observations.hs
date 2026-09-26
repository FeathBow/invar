{-# LANGUAGE OverloadedStrings #-}

module Observations (observations) where

import Control.Monad (forM_)
import Data.Aeson (Value (..), object, toJSON, (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString.Char8 qualified as Bytes
import Hedgehog
import Invar.Learn.Report qualified as Report
import Updates (alter, change, field, setup, wire)
import Workloads (array, replace)

observations :: Group
observations = Group "Conditional update observations" [("report identity covers the entire supplied log", once identity), ("selected calls require one ordered pair and exact binding", once selected), ("numerical request and sample schemas are complete", once schema), ("cohort membership and response boundaries are mandatory", once membership), ("finite numerical coefficients and actual word domains are mandatory", once numbers), ("logical pairing retains all consumed inputs", once paired), ("requests an earlier learning program recorded are refused at its boundary", once boundary), ("duplicate keys and malformed unrelated events fail", once ambiguous), ("integral JSON syntax has core numeric semantics", once integral)]
  where
    once = withTests 1 . property

fixture :: PropertyT IO [Value]
fixture = snd <$> setup

accepted :: [Value] -> PropertyT IO Report.Report
accepted = evalEither . Report.admit selectedCall . wire

selectedCall :: (Num value) => value
selectedCall = 7

identity :: PropertyT IO ()
identity = do
    events <- fixture
    original <- accepted events
    observed <- accepted (object ["stage" .= String "unrelated"] : events)
    assert (Report.logDigest original /= Report.logDigest observed)
    Report.request original === Report.request observed
    Report.invocation original === Report.invocation observed
    minimal <- accepted (drop 1 events)
    Report.result minimal === Report.result original
    noNewline <- evalEither (Report.admit selectedCall (Bytes.init (wire events)))
    Report.request noNewline === Report.request original

selected :: PropertyT IO ()
selected = do
    events <- fixture
    forM_ [[], take 2 events, reverse events, events ++ drop 1 events, events ++ [at 2 events]] reject
    forM_ ["call", "attempt", "instance"] $ \axis -> do
        let bound = field "binding" (at 1 events)
        reject (alter 2 (change "binding" (change axis (Number 99) bound)) events)
        forM_ [Bool True, Number (-1), Number 0.5, Null] $ \value ->
            reject (map (change "binding" (change axis value bound)) events)
    let bound = field "binding" (at 1 events)
    reject (map (change "binding" (change "extra" Null bound)) events)
    reject (alter 1 (change "program" (String "")) events)

schema :: PropertyT IO ()
schema = do
    events <- fixture
    original <- accepted events
    case Report.request original of
        Object fields -> forM_ (Fields.keys fields) $ \name -> reject (requests (omit name) events)
        _ -> failure
    case array (field "samples" (Report.request original)) of
        Object fields : _ -> forM_ (Fields.keys fields) $ \name -> reject (requests (sample (omit name)) events)
        _ -> failure
    reject (requests (change "extra" Null) events)
    reject (requests (sample (change "extra" Null)) events)
    forM_ ["learning_rate", "betas", "epsilon", "weight_decay"] $ \name ->
        reject (requests (\value -> change "optimizer" (omit name (field "optimizer" value)) value) events)

membership :: PropertyT IO ()
membership = do
    events <- fixture
    forM_ [change "samples" (toJSON ([] :: [Value])), \value -> change "samples" (toJSON (array (field "samples" value) ++ array (field "samples" value))) value, change "order" (toJSON ([] :: [Value])), sample (change "sample" (String "")), sample (change "group" (String "foreign")), sample (change "prompt_length" (Number 0)), sample (change "limit" (Number 0)), sample (change "tokens" (toJSON [Number (-1)])), sample (change "behavior_bits" (toJSON ([] :: [Value]))), sample (change "truncated" (Number 1))] $ \mutation ->
        reject (requests mutation events)

numbers :: PropertyT IO ()
numbers = do
    events <- fixture
    forM_ [("epsilon", Number 0), ("epsilon", Number 1), ("penalty", Number (-1)), ("delta", Number 0), ("penalty", Number (10 ^ overflowExponent))] $ \(name, value) -> reject (requests (change name value) events)
    forM_ [("seed", Bool True), ("seed", Number 0.5), ("temperature", Number 0), ("reward", Number (10 ^ overflowExponent)), ("advantage_bits", Number nanWord), ("advantage_bits", Number wordLimit), ("behavior_bits", toJSON [Number positiveOne]), ("behavior_bits", toJSON [Bool False]), ("behavior_bits", toJSON [Number nanWord])] $ \(name, value) -> reject (requests (sample (change name value)) events)
    forM_ [("betas", toJSON [Number 1, Number 0]), ("betas", toJSON [Number 0]), ("epsilon", Number 0), ("learning_rate", Number (-1))] $ \(name, value) -> reject (requests (\input -> change "optimizer" (change name value (field "optimizer" input)) input) events)
  where
    overflowExponent = 400 :: Int
    nanWord = 2143289344
    positiveOne = 1065353216
    wordLimit = 4294967296

paired :: PropertyT IO ()
paired = do
    events <- fixture
    original <- accepted events
    let reordered value = change "samples" (toJSON (reverse (array (field "samples" value)))) value
    delivered <- accepted (requests reordered events)
    Report.paired original delivered === Right ()
    reject (alter 2 (\event -> change "request" (reordered (field "request" event)) event) events)
    changed <- accepted (requests (sample (change "seed" (Number 999))) events)
    rejected (Report.paired original changed)
    program <- accepted (alter 1 (change "program" (String "another program")) events)
    rejected (Report.paired original program)

boundary :: PropertyT IO ()
boundary = do
    events <- fixture
    let earlier = requests (sample (omit "reference_bits")) events
        failure' = either Just (const Nothing) . Report.admit selectedCall . wire
    failure' (alter 1 (change "program" (String "earlier program")) earlier) === Just ("Error in $: " ++ Report.recordedElsewhere)
    current <- evalMaybe (failure' earlier)
    assert (current /= "Error in $: " ++ Report.recordedElsewhere)

ambiguous :: PropertyT IO ()
ambiguous = do
    events <- fixture
    forM_ [wire events <> "{", "null\n" <> wire events, "{\"stage\":\"x\",\"stage\":\"x\"}\n" <> wire events, replace "\"seed\":" "\"seed\":0,\"seed\":" (wire events)] $ \encoded ->
        case Report.admit selectedCall encoded of
            Left _ -> success
            Right value -> annotateShow (Report.describe value) >> failure

integral :: PropertyT IO ()
integral = do
    events <- fixture
    original <- accepted events
    actual <- evalEither (Report.admit selectedCall (replace "\"call\":7" "\"call\":7.0" (wire events)))
    Report.invocation actual === Report.invocation original

requests :: (Value -> Value) -> [Value] -> [Value]
requests mutation = alter 1 update . alter 2 update
  where
    update event = change "request" (mutation (field "request" event)) event

sample :: (Value -> Value) -> Value -> Value
sample mutation value = change "samples" (toJSON (alter 0 mutation (array (field "samples" value)))) value

omit :: Key -> Value -> Value
omit name (Object fields) = Object (Fields.delete name fields)
omit _ value = value

reject :: [Value] -> PropertyT IO ()
reject events = case Report.admit selectedCall (wire events) of
    Left _ -> success
    Right value -> annotateShow (Report.describe value) >> failure

rejected :: Either String () -> PropertyT IO ()
rejected (Left _) = success
rejected (Right ()) = failure

at :: Int -> [value] -> value
at index values = case drop index values of
    value : _ -> value
    [] -> error "Missing fixed observation fixture event"
