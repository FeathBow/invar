{-# LANGUAGE OverloadedStrings #-}

module InferenceObservations (inferenceObservations, fixture, bound, admitted) where

import BatchCalls qualified as Batch
import Calls (change, field, request, setup, wire)
import Control.Monad (forM_)
import Data.Aeson (Value (..), eitherDecodeStrict, object, toJSON, (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Either (isLeft)
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8)
import Data.Word (Word32)
import Hedgehog
import Invar.Artifact qualified as Artifact
import Invar.Infer qualified as Infer
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Observation qualified as Observation
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Result qualified as Result
import Invar.Infer.Session qualified as Session
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Numerical qualified as Numerical
import Invar.Policy qualified as Policy
import Invar.Score qualified as Score
import Invar.Spec.Invocation qualified as V
import Store (workspace)
import System.Exit (ExitCode (..))
import System.FilePath ((</>))
import Workloads (everywhere, replace)

inferenceObservations :: Group
inferenceObservations = Group "Complete inference observations" [("complete observations retain bound results and exact source identity", once matching), ("repeated missing reordered and trailing records fail", once complete), ("loads programs bindings and model images must correspond", once correspondence), ("strict result field and raw JSON schemas apply", once malformed), ("behavior zero signs retain actual floating literals", once signedZero), ("integral spellings of counts seeds and bindings are admitted as integers", once integral), ("a log is replayed under the policy it was declared with, and another model is refused", once described), ("batch members retain the complete source through numerical and score inputs", once batchMatching), ("complete batch sources require loading and one complete framed execution", once batchBoundaries), ("a batch is admitted only under its whole declaration: absent, reused, repeated, reordered and mismatched members are refused", once batchPeers)]
  where
    once = withTests 1 . property

bound :: V.Binding
bound = V.Binding (V.CallId 7) (V.AttemptId 11) (V.Instance 13)

fixture :: PropertyT IO [Value]
fixture = do
    (_, events) <- setup
    case events of
        [loaded, consumed, output] -> pure [object ["stage" .= String "load", "cpu_seconds" .= Number 1], change "model" (String "test-model") (change "revision" (String "test-revision") loaded), consumed, object ["stage" .= String "inference", "cpu_seconds" .= Number 1], output]
        _ -> failure

admitted :: Infer.Plan -> V.Binding -> ByteString -> Either String Observation.Report
admitted planned selected encoded = do
    call <- either (Left . show) Right (Call.prepare selected planned)
    trajectories <- either (Left . show) Right (Replay.standalone Session.Single (Session.Declaration [call] Nothing) ExitSuccess encoded)
    case trajectories of
        [single] -> pure (Observation.view encoded single)
        _ -> Left "Expected one admitted inference"

observe :: ByteString -> Either String Observation.Report
observe encoded = do
    planned <- either (Left . show) Right (Infer.prepare request)
    admitted planned bound encoded

batched :: [Call.Call] -> ByteString -> Either String [Observation.Report]
batched declared encoded = either (Left . show) (Right . map (Observation.view encoded)) (Replay.standalone Session.Batched (Session.Declaration declared Nothing) ExitSuccess encoded)

matching :: PropertyT IO ()
matching = do
    events <- fixture
    original <- evalEither (observe (wire events))
    Result.response (Observation.result original) === "#### 12"
    Observation.binding original === bound
    field "adapter" (Observation.describe original) === toJSON (Infer.artifact request)
    field "tokenizer" (Observation.describe original) === toJSON (Infer.tokenizer request)
    field "request" (Observation.describe original) === object ["prompt" .= Infer.prompt request, "tokens" .= Infer.tokens request, "temperature" .= Infer.temperature request, "seed" .= Infer.seed request]
    changed <- evalEither (observe (wire (alter 0 (change "cpu_seconds" (Number 3)) events)))
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
    forM_ [1, 2, 4] $ \index ->
        forM_ ["call", "attempt", "instance"] $ \axis ->
            reject (wire (alter index (\value -> change "binding" (change axis (Number 99) (field "binding" value)) value) events))
    forM_ [1, 2] $ \index ->
        reject (wire (alter index (\value -> change "load" (change "program" (String "wrong") (field "load" value)) value) events))
    reject (wire (alter 2 (change "program" (String "wrong")) events))
    forM_ ["artifact", "profile"] $ \axis ->
        reject (wire (alter 1 (\value -> change "image" (change axis (String "wrong") (field "image" value)) value) events))
    forM_ ["model", "revision"] $ \axis -> reject (wire (alter 1 (change axis (String "")) events))
    forM_ ["tokenizer", "base", "assembly"] $ \axis ->
        forM_ [1, 2, 4] $ \index -> reject (wire (alter index (change axis (String (Text.replicate 64 "b"))) events))

malformed :: PropertyT IO ()
malformed = do
    events <- fixture
    reject (replace "\"call\":7" "\"call\":7,\"call\":7" (wire events))
    forM_ [Null, Bool True, Number 0.5, Number (-1)] $ \value -> reject (wire (alter 4 (change "prompt_length" value) events))
    forM_ ["tokens", "behavior", "behavior_bits", "request", "binding"] $ \key ->
        reject (wire (alter 4 (omit key) events))
    reject (wire (alter 4 (change "extra" Null) events))
    reject (wire (alter 2 (change "extra" Null) events))

signedZero :: PropertyT IO ()
signedZero = do
    events <- fixture
    let negativeWord = 0x80000000 :: Word32
        quarterWord = 0xbe800000 :: Word32
        withBits word = wire (alter 4 (change "behavior_bits" (toJSON [word, quarterWord])) events)
        spelling text word = replace "[-0.5,-0.25]" text (withBits word)
    forM_ [("[-0.0,-0.25]", negativeWord), ("[0.0,-0.25]", 0), ("[0,-0.25]", 0), ("[-0,-0.25]", 0)] $ \(text, word) -> do
        actual <- evalEither (observe (spelling text word))
        Result.behaviorBits (Observation.result actual) === [word, quarterWord]
    forM_ [("[0.0,-0.25]", negativeWord), ("[-0.0,-0.25]", 0), ("[-0,-0.25]", negativeWord)] $ \(text, word) -> reject (spelling text word)

integral :: PropertyT IO ()
integral = do
    events <- fixture
    original <- evalEither (observe (wire events))
    let spellings = [("\"seed\":17", "\"seed\":17.0"), ("\"call\":7", "\"call\":7.0"), ("[1,2,3]", "[1.0,2.0,3.0]"), ("\"prompt_length\":1", "\"prompt_length\":1.0")]
    actual <- evalEither (observe (foldr (uncurry everywhere) (wire events) spellings))
    Observation.result actual === Observation.result original
    Observation.binding actual === bound

described :: PropertyT IO ()
described = do
    events <- fixture
    plain <- evalEither (Infer.prepare request)
    let description model = Policy.describe (model, "test-revision") (Infer.artifact request, Infer.tokenizer request, Infer.base request, Infer.assembly request)
        replayed planned = do
            call <- either (Left . show) Right (Call.prepare bound planned)
            either (Left . show) Right (Replay.standalone Session.Single (Session.Declaration [call] Nothing) ExitSuccess (wire events))
    declaredPlan <- evalEither (description "test-model" >>= either (Left . show) Right . (`Infer.bindPolicy` plain))
    other <- evalEither (description "other-model" >>= either (Left . show) Right . (`Infer.bindPolicy` plain))
    underPolicy <- evalEither (replayed declaredPlan)
    underMaterialization <- evalEither (replayed plain)
    map Trajectory.binding underPolicy === [bound]
    assert (map Trajectory.digest underPolicy /= map Trajectory.digest underMaterialization)
    assert (isLeft (replayed other))

batchFixture :: PropertyT IO ([Value], [Value], [Call.Call])
batchFixture = do
    original <- drop 1 <$> fixture
    planned <- evalEither (Infer.prepare request)
    own <- evalEither (Call.prepare bound planned)
    let unmeasured = take 2 original ++ drop 3 original
        measured member = take 2 member ++ take 1 (drop 2 original) ++ drop 2 member
    (peerCall, peer) <- Batch.prepared planned unmeasured 23 Nothing
    pure (original, alter 3 (change "text" (String "peer result")) (measured peer), [own, peerCall])

withLoad :: [Value] -> ByteString
withLoad member = wire (object ["stage" .= String "load", "cpu_seconds" .= Number 1] : member)

batchFrames :: [[Value]] -> [Value]
batchFrames members =
    [ object ["stage" .= String "load", "cpu_seconds" .= Number 1]
    , frame "consumed" (map (wire . take 2) members)
    , object ["stage" .= String "inference", "cpu_seconds" .= Number 2]
    , frame "result" (map (wire . pure . last) members)
    ]
  where
    frame stage values = object ["stage" .= (stage :: Text.Text), "format" .= String "invar-inference-batch-v1", "calls" .= map decodeUtf8 values]

batchMatching :: PropertyT IO ()
batchMatching = do
    (original, peer, declared) <- batchFixture
    planned <- evalEither (Infer.prepare request)
    root <- workspace
    let events = batchFrames [original, peer]
        encoded = wire events
        peerBound = V.Binding (V.CallId 23) (V.AttemptId 23) (V.Instance 23)
        source = root </> "complete-batch.jsonl"
    evalIO (Bytes.writeFile source encoded)
    digest <- evalIO (Artifact.identity "complete batch fixture" source)
    members <- evalEither (batched declared encoded)
    changedMembers <- evalEither (batched declared (wire (alter 0 (change "cpu_seconds" (Number 3)) events)))
    map Observation.binding members === [bound, peerBound]
    forM_ (zip3 members changedMembers [(bound, original), (peerBound, peer)]) $ \(observed, changed, (selected, standalone)) -> do
        expected <- evalEither (admitted planned selected (withLoad standalone))
        Observation.binding observed === selected
        Observation.result observed === Observation.result expected
        Observation.logDigest observed === digest
        Observation.result changed === Observation.result observed
        assert (Observation.logDigest changed /= Observation.logDigest observed)
        scoring <- evalEither (Score.prepare 0 observed planned)
        inspection <- evalEither (eitherDecodeStrict (Score.sourceInspection scoring))
        inspection === Observation.describe observed
    compared <- evalEither (Numerical.observe (Numerical.BoundRun (Numerical.Run planned bound 0 (withLoad original) Nothing) (Numerical.Run planned bound 0 encoded (Just declared))))
    Numerical.tokensEqual (Numerical.path compared) === True
    Numerical.behaviorBitsEqual (Numerical.path compared) === True

batchBoundaries :: PropertyT IO ()
batchBoundaries = do
    (original, peer, declared) <- batchFixture
    let events = batchFrames [original, peer]
        rejected = refuse declared
    _ <- evalEither (batched declared (wire events))
    forM_ [[], drop 1 events, take 3 events, reverse events, events ++ events, events ++ take 1 events] (rejected . wire)
    forM_ [0 .. length events - 1] $ \index -> do
        rejected (wire (take index events ++ drop (index + 1) events))
        rejected (wire (take index events ++ [events !! index] ++ drop index events))
        rejected (wire (alter index (change "phase" (String "other")) events))
    forM_ [1, 3] $ \index -> do
        rejected (wire (alter index (change "format" (String "unknown")) events))
        rejected (wire (alter index (omit "format") events))
    forM_ [0, 2] $ \index -> rejected (wire (alter index (change "cpu_seconds" (Number (-1))) events))
    rejected (Bytes.init (wire events))

batchPeers :: PropertyT IO ()
batchPeers = do
    (original, peer, declared) <- batchFixture
    planned <- evalEither (Infer.prepare request)
    let events = batchFrames [original, peer]
        absent = V.Binding (V.CallId 99) (V.AttemptId 99) (V.Instance 99)
        rejected = refuse declared
        run selected = Numerical.Run planned selected 0 (wire events) (Just declared)
    reseeded <- evalEither (Infer.prepare request {Infer.seed = 18})
    _ <- evalEither (Numerical.admit Numerical.Reference (run bound))
    case Numerical.admit Numerical.Reference (run bound) {Numerical.planned = reseeded} of
        Left _ -> success
        Right _ -> failure
    case Numerical.admit Numerical.Reference (run absent) of
        Left _ -> success
        Right _ -> failure
    refuse (reverse declared) (wire events)
    refuse (take 1 declared ++ take 1 declared) (wire events)
    refuse (take 1 declared) (wire events)
    rejected (wire (batchFrames [original, original]))
    forM_ [0, 1, 3] $ \index ->
        forM_ ["call", "attempt", "instance"] $ \axis -> do
            let wrong = alter index (\value -> change "binding" (change axis (Number 99) (field "binding" value)) value) peer
            rejected (wire (batchFrames [original, wrong]))
    let reversed = last (batchFrames [peer, original])
        missing = last (batchFrames [original])
    rejected (wire (take 3 events ++ [reversed]))
    rejected (wire (take 3 events ++ [missing]))

alter :: Int -> (value -> value) -> [value] -> [value]
alter selected changeValue = zipWith (\index value -> if index == selected then changeValue value else value) [0 ..]

omit :: Key -> Value -> Value
omit key (Object fields) = Object (Fields.delete key fields)
omit _ value = value

refuse :: [Call.Call] -> ByteString -> PropertyT IO ()
refuse declared encoded = case batched declared encoded of
    Left _ -> success
    Right reports -> annotateShow (map Observation.describe reports) >> failure

reject :: ByteString -> PropertyT IO ()
reject encoded = case observe encoded of
    Left _ -> success
    Right report -> annotateShow (Observation.describe report) >> failure
