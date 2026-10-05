{-# LANGUAGE OverloadedStrings #-}

module Scores (scores, fixtureFor, fixtureWith, changeObservation, nativeExecutionValue) where

import Calls (change, field, request, wire)
import Control.Monad (forM_)
import Data.Aeson (Value (..), eitherDecodeStrict, object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString.Char8 qualified as Bytes
import Data.Text qualified as Text
import Data.Word (Word32)
import Hedgehog
import InferenceObservations qualified as Source
import Invar.Infer qualified as Infer
import Invar.Infer.Observation qualified as Inference
import Invar.Infer.Result qualified as Result
import Invar.Score qualified as Score
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as Load
import Workloads (replace)

scores :: Group
scores =
    Group
        "Bound cached path scores"
        [ ("actual source request is required before constructing a score", once sourceInputs)
        , ("checked load and consumption produce a distinct score completion", once admitted)
        , ("path ratios preserve finite arithmetic signed zero and zero support", once arithmetic)
        , ("score bindings programs and materializations cannot be relabeled", once bindings)
        , ("source path probability role and execution semantics must match", once correspondence)
        , ("malformed nonfinite and incomplete score reports fail", once malformed)
        , ("completion must retain its authorized prefix and successful process", once lifecycle)
        , ("native scoring preserves its own execution and row accounting contract", once nativeExecution)
        ]
  where
    once = withTests 1 . property

fixture :: PropertyT IO (Score.Call, [Value])
fixture = do
    original <- Source.fixture
    fixtureFor original request

fixtureFor :: [Value] -> Infer.Request -> PropertyT IO (Score.Call, [Value])
fixtureFor = fixtureWith Right

fixtureWith :: (Score.Plan -> Either Score.Error Score.Plan) -> [Value] -> Infer.Request -> PropertyT IO (Score.Call, [Value])
fixtureWith configure original target = do
    sourcePlan <- evalEither (Infer.prepare request)
    source <- evalEither (Source.admitted sourcePlan Source.bound (wire original))
    targetPlan <- evalEither (Infer.prepare target)
    planned <- evalEither (Score.prepare 0 source targetPlan >>= configure)
    call <- evalEither (Score.bind (V.Binding (V.CallId 20) (V.AttemptId 21) (V.Instance 22)) planned)
    envelope <- evalEither (eitherDecodeStrict (Score.input call))
    sourceLoaded <- case filter ((== String "loaded_adapter") . field "stage") original of
        Object loaded : _ -> pure (Object (Fields.delete "scope" loaded))
        _ -> failure
    let bound = field "binding" envelope
        image = Infer.image target
        loaded =
            foldr
                (uncurry change)
                sourceLoaded
                [("binding", bound), ("load", field "load" envelope), ("image", object ["artifact" .= Bytes.unpack (Load.artifact image), "profile" .= Bytes.unpack (Load.profile image)]), ("requested", toJSON (Infer.artifact target)), ("consumed", toJSON (Infer.artifact target)), ("tokenizer", toJSON (Infer.tokenizer target)), ("base", toJSON (Infer.base target)), ("assembly", toJSON (Infer.assembly target))]
        consumed = change "stage" (String "consumed") envelope
        sourceResult = Inference.result source
        (prefix, response) = splitAt (fromIntegral (Result.promptLength sourceResult)) (Result.tokens sourceResult)
        observed =
            object
                [ "format" .= String "invar-cached-path-score-v1"
                , "role" .= String "cached_behavior_cross_score"
                , "use_admission" .= String "not_evaluated"
                , "source_inspection_sha256" .= Score.sourceDigest planned
                , "source" .= Inference.describe source
                , "target" .= object ["adapter" .= Infer.artifact target, "tokenizer" .= Infer.tokenizer target, "base" .= Infer.base target, "assembly" .= Infer.assembly target, "model" .= field "model" sourceLoaded, "revision" .= field "revision" sourceLoaded, "numerics" .= String "test-profile"]
                , "request" .= field "request" envelope
                , "prefix_tokens" .= prefix
                , "response_tokens" .= response
                , "log_probability_bits" .= Result.behaviorBits sourceResult
                , "probability" .= object ["role" .= String "behavior", "log_base" .= String "e", "representation" .= String "F32 words", "zero_support_word" .= (0xff800000 :: Word32), "temperature" .= Infer.temperature request, "mask" .= String "none", "top_k" .= String "disabled", "top_p" .= String "disabled"]
                , "execution" .= object ["engine" .= String "mlx_lm.generate.BatchGenerator", "sampling" .= object ["batch_size" .= (1 :: Int), "prefill_step" .= (2 :: Int)], "cache_origin" .= String "fresh native caches; no reference cache input", "unused_native_lookahead_draws" .= (1 :: Int), "truncated" .= Result.truncated sourceResult]
                , "implementation" .= object ["sources_sha256" .= object ["unit-fixture" .= Text.replicate 64 "a"], "packages" .= object ["fixture" .= String "v1"]]
                ]
        result = object ["stage" .= String "score_result", "binding" .= bound, "observation" .= observed]
    pure (call, [loaded, consumed, object ["stage" .= String "verify_before"], object ["stage" .= String "cross_score"], object ["stage" .= String "verify_after"], result])

sourceInputs :: PropertyT IO ()
sourceInputs = do
    original <- Source.fixture
    planned <- evalEither (Infer.prepare request)
    source <- evalEither (Source.admitted planned Source.bound (wire original))
    case Score.prepare 3 source planned of
        Left (Score.SourceProcessFailed 3) -> success
        _ -> failure
    forM_ [request {Infer.prompt = "different"}, request {Infer.tokens = 9}, request {Infer.temperature = 1.1}, request {Infer.seed = 999}, request {Infer.tokenizer = replicate 64 'e'}] $ \input -> do
        target <- evalEither (Infer.prepare input)
        case Score.prepare 0 source target of
            Left (Score.Incompatible _) -> success
            _ -> failure

admitted :: PropertyT IO ()
admitted = do
    (call, events) <- fixture
    report <- evalEither (Score.admit call 0 (wire events))
    Score.logRatio report === Score.Finite 0
    V.completedBinding (Score.completion report) === Score.binding call
    field "use_admission" (Score.describe report) === String "not_evaluated"
    (registry, permit) <- evalEither (Score.authorize Load.empty call (wire (take 2 events)))
    length (Load.active registry) === 1
    after <- evalEither (Score.observe permit (wire events))
    after === report
    declared <- evalEither (eitherDecodeStrict (Score.input call))
    granted <- evalEither (eitherDecodeStrict (Score.permission permit))
    field "program" granted === field "program" declared

arithmetic :: PropertyT IO ()
arithmetic = do
    (call, events) <- fixture
    let bits encoded = changeObservation (change "log_probability_bits" (toJSON (encoded :: [Word32]))) events
    forM_ [([0xbe800000, 0xbf000000], Score.Finite 0), ([0xbe000000, 0xbf000000], Score.Finite (-(1 / 8))), ([0x80000000, 0x80000000], Score.Finite (-(3 / 4))), ([0xbf000000, 0xff800000], Score.PositiveInfinity)] $ \(encoded, expected) -> do
        observed <- evalEither (Score.admit call 0 (wire (bits encoded)))
        Score.logRatio observed === expected
        Score.probabilityWords observed === encoded

bindings :: PropertyT IO ()
bindings = do
    (call, events) <- fixture
    forM_ [0, 1, 5] $ \index ->
        forM_ ["call", "attempt", "instance"] $ \axis -> reject call (alter index (\value -> change "binding" (change axis (Number 99) (field "binding" value)) value) events)
    forM_ [0, 1] $ \index -> reject call (alter index (\value -> change "load" (change "program" (String "other") (field "load" value)) value) events)
    reject call (alter 1 (change "program" (String "other")) events)
    forM_ ["tokenizer", "base", "assembly"] $ \axis ->
        forM_ [0, 1] $ \index -> do
            let changed = String (Text.replicate 64 "0")
            forM_ (take 1 (drop index events)) $ \original -> assert (field axis original /= changed)
            reject call (alter index (change axis changed) events)
    forM_ ["artifact", "profile"] $ \axis -> reject call (alter 0 (\value -> change "image" (change axis (String "wrong") (field "image" value)) value) events)

correspondence :: PropertyT IO ()
correspondence = do
    (call, events) <- fixture
    forM_ ["source_inspection_sha256", "source", "target", "request", "prefix_tokens", "response_tokens"] $ \key -> reject call (changeObservation (change key Null) events)
    forM_ [("role", String "unscaled"), ("temperature", Number 2), ("top_k", String "8")] $ \(key, value) ->
        reject call (changeObservation (\body -> change "probability" (change key value (field "probability" body)) body) events)
    forM_ [("cache_origin", String "reference cache"), ("unused_native_lookahead_draws", Number 0), ("truncated", Bool False)] $ \(key, value) ->
        reject call (changeObservation (\body -> change "execution" (change key value (field "execution" body)) body) events)
    reject call (alter 1 (change "source_inspection" (String "{}")) events)

malformed :: PropertyT IO ()
malformed = do
    (call, events) <- fixture
    forM_ [[0x7f800000, 0], [0x7fc00000, 0], [0x3f800000, 0], [0x100000000, 0], [-1, 0], []] $ \encoded ->
        reject call (changeObservation (change "log_probability_bits" (toJSON (encoded :: [Integer]))) events)
    forM_ [[], reverse events, events ++ events, take 5 events, events ++ [object ["stage" .= String "load"]]] (reject call)
    forM_ [0 .. length events - 1] $ \index -> reject call (take index events ++ drop (index + 1) events)
    rejectBytes call (Bytes.init (wire events))
    rejectBytes call (replace "\"call\":20" "\"call\":20,\"call\":20" (wire events))
    reject call (object ["stage" .= String "loading", "phase" .= String "result"] : events)

lifecycle :: PropertyT IO ()
lifecycle = do
    (call, events) <- fixture
    Score.admit call 7 (wire events) === Left (Score.ProcessFailed 7)
    (_, permit) <- evalEither (Score.authorize Load.empty call (wire (take 2 events)))
    case Score.observe permit (" " <> wire events) of
        Left (Score.Protocol _) -> success
        _ -> failure

nativeExecution :: PropertyT IO ()
nativeExecution = do
    (call, events) <- fixture
    let execution = nativeExecutionValue
        scored = changeObservation (change "execution" execution) events
        changed key value = changeObservation (change "execution" (change key value execution)) events
    report <- evalEither (Score.admit call 0 (wire scored))
    Score.logRatio report === Score.Finite 0
    (_, permit) <- evalEither (Score.authorize Load.empty call (wire (take 2 scored)))
    evalEither (Score.observe permit (wire scored)) >>= (=== report)
    zero <- evalEither (Score.admit call 0 (wire (changeObservation (change "log_probability_bits" (toJSON ([0xff800000, 0] :: [Word32]))) scored)))
    Score.logRatio zero === Score.PositiveInfinity
    forM_ [("engine", String "unknown"), ("engine", String "mlx_lm.generate.BatchGenerator"), ("cache_origin", String "reference cache"), ("path_control", String "allowed-token whitelist"), ("native_sample_rows", Number 4), ("native_sample_rows", Bool True), ("ignored_prefill_rows", Number (-1)), ("ignored_prefill_rows", Number 3.5), ("truncated", Bool False), ("unused_native_lookahead_draws", Number 1)] $ \(key, value) -> reject call (changed key value)
    case execution of
        Object fields -> reject call (changeObservation (change "execution" (Object (Fields.delete "path_control" fields))) events)
        _ -> failure

nativeExecutionValue :: Value
nativeExecutionValue = object ["engine" .= String "vllm.v1.worker.gpu_model_runner.GPUModelRunner", "cache_origin" .= String "target-owned native request caches; no source cache input", "path_control" .= String "replace sampled ids after native probability calculation", "native_sample_rows" .= (5 :: Int), "ignored_prefill_rows" .= (3 :: Int), "truncated" .= True]

alter :: Int -> (a -> a) -> [a] -> [a]
alter selected modify = zipWith (\index value -> if index == selected then modify value else value) [0 ..]

changeObservation :: (Value -> Value) -> [Value] -> [Value]
changeObservation modify = alter 5 (\value -> change "observation" (modify (field "observation" value)) value)

reject :: Score.Call -> [Value] -> PropertyT IO ()
reject call = rejectBytes call . wire

rejectBytes :: Score.Call -> Bytes.ByteString -> PropertyT IO ()
rejectBytes call encoded = case Score.admit call 0 encoded of
    Left _ -> success
    Right report -> annotateShow (Score.describe report) >> failure
