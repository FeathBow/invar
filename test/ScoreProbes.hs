{-# LANGUAGE OverloadedStrings #-}

module ScoreProbes (scoreProbes, fixtureFor) where

import Calls (change, field, request, wire)
import Control.Monad (forM_)
import Data.Aeson (Value (..), eitherDecodeStrict, object, toJSON, (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.Word (Word32)
import Hedgehog
import InferenceObservations qualified as Source
import Invar.Infer qualified as Infer
import Invar.Infer.Observation qualified as Inference
import Invar.Score qualified as Score
import Invar.Spec.Load qualified as Load
import Numeric.Natural (Natural)
import Scores qualified

scoreProbes :: Group
scoreProbes =
    Group
        "Bound full-vocabulary probes"
        [ ("probe selection is checked against the actual response", once selection)
        , ("probe source and steps are frozen before consumption", once binding)
        , ("complete vectors become an opaque observation without a KL claim", once admitted)
        , ("complete vocabulary and selected-step inventory are required", once inventory)
        , ("nonfinite negative and out-of-range masses are rejected", once masses)
        , ("zero support agrees with the selected-token path score", once support)
        , ("cost records must match actual execution diagnostics", once costs)
        , ("separate native log words do not determine represented support", once nativeSupport)
        , ("native probes require explicit measurement and worker resource contracts", once nativeContract)
        ]
  where
    once = withTests 1 . property

fixture :: [Natural] -> PropertyT IO (Score.Call, [Value])
fixture steps = do
    original <- Source.fixture
    fixtureFor steps original request

fixtureFor :: [Natural] -> [Value] -> Infer.Request -> PropertyT IO (Score.Call, [Value])
fixtureFor steps original target = do
    (call, plain) <- Scores.fixtureWith (Score.withProbe steps) original target
    envelope <- evalEither (eitherDecodeStrict (Score.input call))
    let recorded = [metric "verify_before", metric "cross_score", metric "verify_after"]
        events = take 2 plain ++ recorded ++ drop 5 plain
        body value =
            foldr
                (uncurry change)
                value
                [ ("format", String "invar-cached-distribution-probe-v1")
                , ("role", String "cached_behavior_full_vocabulary")
                , ("source_inspection", field "source_inspection" envelope)
                , ("measurements", toJSON recorded)
                ,
                    ( "full_vocabulary"
                    , object
                        [ "steps" .= steps
                        , "vocabulary" .= (4 :: Int)
                        , "coordinates" .= String "output token ids 0..vocabulary-1"
                        , "representation" .= String "F32 probability words"
                        , "snapshots" .= [snapshot step (replicate 4 0x3e800000) | step <- steps]
                        , "raw_payload_bytes" .= (length steps * 4 * 4)
                        ]
                    )
                ]
    pure (call, Scores.changeObservation body events)

metric :: String -> Value
metric name = object ["stage" .= name, "seconds" .= (0.1 :: Double), "allocator" .= String "mlx", "peak_active" .= (64 :: Int), "cache_end" .= (0 :: Int)]

snapshot :: Natural -> [Word32] -> Value
snapshot step encodedMasses = object ["step" .= step, "probability_bits" .= encodedMasses]

selection :: PropertyT IO ()
selection = do
    original <- Source.fixture
    planned <- evalEither (Infer.prepare request)
    source <- evalEither (Inference.admit planned Source.bound (wire original))
    base <- evalEither (Score.prepare 0 source planned)
    forM_ [[], [0, 0], [1, 0], [2], [0, 2]] $ \steps -> case Score.withProbe steps base of
        Left (Score.Incompatible _) -> success
        _ -> failure
    forM_ [[0], [1], [0, 1]] $ \steps -> evalEither (Score.withProbe steps base) >> success

binding :: PropertyT IO ()
binding = do
    (call, events) <- fixture [0, 1]
    (other, _) <- fixture [0]
    original <- Source.fixture
    (plain, _) <- Scores.fixtureFor original request
    declared <- evalEither (eitherDecodeStrict (Score.input call))
    ordinary <- evalEither (eitherDecodeStrict (Score.input plain))
    assert (field "program" declared /= field "program" ordinary)
    assert (Score.input call /= Score.input other)
    reject other events
    reject plain events
    let changed = alter 1 (change "probe_steps" (toJSON [0 :: Int])) events
    case Score.authorize Load.empty call (wire (take 2 changed)) of
        Left (Score.Protocol _) -> success
        _ -> failure
    reject call (Scores.changeObservation (change "source_inspection" (String "{}")) events)

admitted :: PropertyT IO ()
admitted = do
    (call, events) <- fixture [0, 1]
    result <- evalEither (Score.admit call 0 (wire events))
    field "strength" (Score.describe result) === String "finite_full_vocabulary_observation"
    field "use_admission" (Score.describe result) === String "not_evaluated"
    field "observation" (Score.describe result) === field "observation" (last events)
    (_, permit) <- evalEither (Score.authorize Load.empty call (wire (take 2 events)))
    checked <- evalEither (Score.observe permit (wire events))
    checked === result
    reject call (Scores.changeObservation (change "execution" Scores.nativeExecutionValue) events)

inventory :: PropertyT IO ()
inventory = do
    (call, events) <- fixture [0, 1]
    forM_ [("steps", toJSON [1, 0 :: Int]), ("steps", toJSON [Bool False, Number 1]), ("vocabulary", Number 5), ("vocabulary", Number 0), ("vocabulary", Bool True), ("raw_payload_bytes", Number 31), ("coordinates", String "other tokenizer"), ("representation", String "logits"), ("snapshots", toJSON ([] :: [Value])), ("snapshots", toJSON [snapshot 1 (replicate 4 0x3e800000), snapshot 0 (replicate 4 0x3e800000)]), ("snapshots", toJSON [snapshot 0 (replicate 4 0x3e800000), snapshot 0 (replicate 4 0x3e800000)])] $ \(key, value) ->
        reject call (vectors (change key value) events)
    reject call (Scores.changeObservation (omit "full_vocabulary") events)
    reject call (Scores.changeObservation (change "format" (String "invar-cached-path-score-v1")) events)

masses :: PropertyT IO ()
masses = do
    (call, events) <- fixture [0]
    forM_ [[0, 0, 0, 0], [0x7fc00000, 0, 0, 0], [0x7f800000, 0, 0, 0], [0xff800000, 0, 0, 0], [0xbf800000, 0, 0, 0], [0x40000000, 0, 0, 0], [0x100000000, 0, 0, 0], [-1, 0, 0, 0], [0x3f800000]] $ \encodedMasses ->
        reject call (vectors (change "snapshots" (toJSON [object ["step" .= (0 :: Int), "probability_bits" .= (encodedMasses :: [Integer])]])) events)

support :: PropertyT IO ()
support = do
    (call, events) <- fixture [0]
    let zero = vectors (change "snapshots" (toJSON [snapshot 0 [0x3f800000, 0, 0, 0]])) events
    reject call zero
    let matching = Scores.changeObservation (change "log_probability_bits" (toJSON [0xff800000, 0xbe800000 :: Word32])) zero
    result <- evalEither (Score.admit call 0 (wire matching))
    Score.logRatio result === Score.PositiveInfinity
    reject call (Scores.changeObservation (change "log_probability_bits" (toJSON [0xff800000, 0xbe800000 :: Word32])) events)
    let signed = vectors (change "snapshots" (toJSON [snapshot 0 [0x3f800000, 0, 0x80000000, 0]])) matching
    signedResult <- evalEither (Score.admit call 0 (wire signed))
    Score.logRatio signedResult === Score.PositiveInfinity

costs :: PropertyT IO ()
costs = do
    (call, events) <- fixture [0]
    reject call (Scores.changeObservation (omit "measurements") events)
    reject call (Scores.changeObservation (change "measurements" (toJSON ([] :: [Value]))) events)
    reject call (alter 3 (change "seconds" (Number 2)) events)
    forM_ [("seconds", Number (-1)), ("seconds", Bool True), ("peak_active", Number 0), ("peak_active", Number 1.5), ("cache_end", Number (-1)), ("allocator", String "unknown")] $ \(key, value) -> do
        let changed = alter 3 (change key value) events
            matching = Scores.changeObservation (change "measurements" (toJSON (take 3 (drop 2 changed)))) changed
        reject call matching

nativeFixture :: PropertyT IO (Score.Call, [Value])
nativeFixture = do
    (call, events) <- fixture [0]
    let distribution = object ["mass_capture" .= String "torch.Tensor.softmax.output/F32/v1", "reported_log" .= String "torch.Tensor.log_softmax.output/F32/v1", "relation" .= String "separately rounded log-softmax and softmax/v1"]
        resources = object ["stage" .= String "cross_score", "seconds" .= (0.1 :: Double), "seconds_scope" .= String "client wait_for_completion/v1", "allocator" .= String "torch.cuda", "workers" .= [nativeWorker]]
        changed = alter 3 (const resources) events
        recorded = toJSON (take 3 (drop 2 changed))
    pure (call, Scores.changeObservation (change "measurements" recorded . change "execution" (change "distribution" distribution Scores.nativeExecutionValue)) changed)

nativeWorker :: Value
nativeWorker = object ["host" .= String "protocol-fixture", "pid" .= (1 :: Int), "device" .= String "cuda:0", "seconds" .= (0.1 :: Double), "peak_allocated" .= (64 :: Int), "peak_reserved" .= (128 :: Int), "scope" .= String "native worker permit-to-completion; loading and serialization excluded/v1"]

nativeSupport :: PropertyT IO ()
nativeSupport = do
    (call, events) <- nativeFixture
    let zero = vectors (change "snapshots" (toJSON [snapshot 0 [0x3f800000, 0, 0, 0]])) events
        finite = Scores.changeObservation (change "log_probability_bits" (toJSON [0xc2f00000, 0xbf000000 :: Word32])) zero
    result <- evalEither (Score.admit call 0 (wire finite))
    Score.logRatio result === Score.Finite (479 / 4)
    field "strength" (Score.describe result) === String "finite_full_vocabulary_observation"
    field "use_admission" (Score.describe result) === String "not_evaluated"
    (_, permit) <- evalEither (Score.authorize Load.empty call (wire (take 2 finite)))
    checked <- evalEither (Score.observe permit (wire finite))
    checked === result

nativeContract :: PropertyT IO ()
nativeContract = do
    (call, events) <- nativeFixture
    let execution modify = Scores.changeObservation (\value -> change "execution" (modify (field "execution" value)) value)
        contract modify value = change "distribution" (modify (field "distribution" value)) value
    reject call (execution (omit "distribution") events)
    forM_ ["mass_capture", "reported_log", "relation"] $ \key ->
        reject call (execution (contract (change key (String "unsupported"))) events)
    forM_ [[], [nativeWorker, nativeWorker]] $ \workers -> rejectNativeCost call (change "workers" (toJSON workers)) events
    forM_ [("pid", Bool True), ("pid", Number 0), ("device", String ""), ("host", String ""), ("seconds", Number (-1)), ("peak_allocated", Number 0), ("peak_reserved", Number 32), ("scope", String "client")] $ \(key, value) ->
        rejectNativeCost call (change "workers" (toJSON [change key value nativeWorker])) events
    forM_ [("allocator", String "mlx"), ("seconds_scope", String "whole process"), ("seconds", Bool True)] $ \(key, value) -> rejectNativeCost call (change key value) events

rejectNativeCost :: Score.Call -> (Value -> Value) -> [Value] -> PropertyT IO ()
rejectNativeCost call modify events = do
    let changed = alter 3 modify events
        matching = Scores.changeObservation (change "measurements" (toJSON (take 3 (drop 2 changed)))) changed
    reject call matching

vectors :: (Value -> Value) -> [Value] -> [Value]
vectors modify = Scores.changeObservation (\value -> change "full_vocabulary" (modify (field "full_vocabulary" value)) value)

alter :: Int -> (a -> a) -> [a] -> [a]
alter selected modify = zipWith (\index value -> if index == selected then modify value else value) [0 ..]

omit :: Key -> Value -> Value
omit key (Object fields) = Object (Fields.delete key fields)
omit _ value = value

reject :: Score.Call -> [Value] -> PropertyT IO ()
reject call events = case Score.admit call 0 (wire events) of
    Left _ -> success
    Right result -> annotateShow (Score.describe result) >> failure
