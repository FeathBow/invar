{-# LANGUAGE OverloadedStrings #-}

module Generation (generation) where

import BatchCalls qualified as Serial
import BatchedProtocol qualified as Batched
import Calls qualified as Fixture
import Control.Monad (foldM, forM_)
import Data.Aeson (Value (..), encode, object, (.=))
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Either (isLeft)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8)
import Data.Word (Word32)
import Hedgehog
import Invar.Canonical qualified as Canonical
import Invar.Infer qualified as I
import Invar.Infer.Invocation qualified as C
import Invar.Infer.Result qualified as R
import Invar.Infer.Session qualified as Session
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Resident.Owner qualified as Owner
import Invar.Spec.Invocation qualified as V
import Invar.Transcript qualified as Transcript
import ResidentFixture qualified as F
import Store (workspace)
import System.Exit (ExitCode (..))

generation :: Group
generation =
    Group
        "Generation admission machine"
        [ ("a serial session sends each request after the previous result, refuses a misplaced or widened load or unload record as it arrives, and admits nothing before a clean exit", once ordered)
        , ("every serial result keeps the zero sign of its behavior values, and its behavior doubles are read from the bits", once signed)
        , ("extra fields, untimed measurements, stray stages, output after the final result, an early or unclean exit and an unterminated record are refused", once refused)
        , ("a profile naming another model than the loaded one is refused in serial and batched sessions", once profiled)
        , ("reference scores are admitted only under the declared reference identity", once referenced)
        , ("a history segment is admitted only when delimited after its final response, and never in a resident session", once delimited)
        , ("one release acknowledgement admits a resident group, and a later interrupted group leaves it admitted", once resident)
        , ("a physical owner refuses reused identities, another model, mixed clocks and a wrong close count", once owned)
        , ("the canonical encoding sorts keys by their bytes and escapes only quotes, backslashes and control characters", once canonical)
        , ("the evidence of a trajectory is fixed byte for byte", once evidenced)
        ]
  where
    once = withTests 1 . property

describe :: Session.Product -> String
describe produced = case produced of
    Session.SendRequest (Session.Request selected) -> "request " ++ show [call | V.Binding (V.CallId call) _ _ <- map callBinding selected]
    Session.Send _ -> "send"
    Session.Close -> "close"
    Session.Admitted trajectories -> "admitted " ++ show [call | V.Binding (V.CallId call) _ _ <- map Trajectory.binding trajectories]
    Session.Owned _ -> "owned"
  where
    callBinding = C.binding

records :: [Value] -> [Session.Input]
records = map Session.Line . Bytes.lines . Fixture.wire

feed :: Session.Session -> [Session.Input] -> Either Session.Error (Session.Session, [Session.Product])
feed initial = foldM (\(current, produced) supplied -> fmap (produced ++) <$> Session.step current supplied) (initial, [])

finished :: Session.Input
finished = Session.Ended (Transcript.Exited ExitSuccess Transcript.Complete)

serial :: PropertyT IO ((C.Call, [Value]), (C.Call, [Value]))
serial = do
    (_, events) <- Fixture.setup
    planned <- evalEither (I.prepare Fixture.request)
    first <- Serial.prepared planned events 0 Nothing
    second <- Serial.prepared planned events 1 (Just 0)
    pure (first, second)

session :: [(C.Call, [Value])] -> [Session.Input]
session requests = concat (zipWith call (True : repeat False) requests)
  where
    call initial (_, events) = records ([Serial.timer "load" | initial] ++ Fixture.reviewPrefix events ++ [Serial.timer "inference"] ++ drop (length (Fixture.reviewPrefix events)) events)

ordered :: PropertyT IO ()
ordered = do
    (first, second) <- serial
    let start = Session.start Session.Serial
        firstInputs = session [first]
        secondInputs = drop (length firstInputs) (session [first, second])
        (readyFirst, completedFirst) = splitAt (length firstInputs - 2) firstInputs
        (readySecond, completedSecond) = splitAt (length secondInputs - 2) secondInputs
    (dispatched, opening) <- evalEither (Session.step start (Session.Dispatched (Session.Declaration [fst first, fst second] Nothing)))
    map describe opening === ["request [0]"]
    (granted, permitted) <- evalEither (feed dispatched readyFirst)
    map describe permitted === ["send"]
    (following, next) <- evalEither (feed granted completedFirst)
    map describe next === ["request [1]"]
    let unload = take 1 secondInputs
        widened = drop (length firstInputs) (session [first, (fst second, [Fixture.change "extra" Null event | event <- take 1 (snd second)] ++ drop 1 (snd second))])
    assert (isLeft (feed dispatched unload))
    assert (isLeft (feed following (unload ++ unload)))
    assert (isLeft (feed following (records [Serial.timer "load"])))
    assert (isLeft (feed following (take 1 widened)))
    (grantedSecond, permittedSecond) <- evalEither (feed following readySecond)
    map describe permittedSecond === ["send"]
    (closing, closed) <- evalEither (feed grantedSecond completedSecond)
    map describe closed === ["close"]
    (_, admitted) <- evalEither (Session.step closing finished)
    map describe admitted === ["admitted [0,1]"]

signed :: PropertyT IO ()
signed = do
    (first, _) <- serial
    let inputs = session [first]
        spelled text word = [replace text word supplied | supplied <- inputs] ++ [finished]
        replace text word (Session.Line raw)
            | "\"result\"" `Bytes.isInfixOf` raw = Session.Line (substitute "[-0.5,-0.25]" text (substitute "[3204448256,3196059648]" ("[" <> word <> ",3196059648]") raw))
        replace _ _ supplied = supplied
        admitted text word = feed (Session.start Session.Serial) (Session.Dispatched (Session.Declaration [fst first] Nothing) : spelled text word)
    forM_ [("[-0.0,-0.25]", "2147483648", True), ("[0.0,-0.25]", "0", False)] $ \(text, word, negative) -> do
        (_, produced) <- evalEither (admitted text word)
        map describe (drop 3 produced) === ["admitted [0]"]
        forM_ [trajectory | Session.Admitted [trajectory] <- produced] $ \trajectory -> do
            map isNegativeZero (Trajectory.behavior trajectory) === [negative, False]
            map isNegativeZero (R.behavior (Trajectory.result trajectory)) === [negative, False]
    forM_ [("[0.0,-0.25]", "2147483648"), ("[-0.0,-0.25]", "0")] $ \(text, word) -> assert (isLeft (admitted text word))

refused :: PropertyT IO ()
refused = do
    (first, _) <- serial
    let (call, events) = first
        valid = session [first] ++ [finished]
        dispatch = Session.Dispatched (Session.Declaration [call] Nothing)
        attempt inputs = feed (Session.start Session.Serial) (dispatch : inputs)
        altered index change = session [(call, [if position == index then change event else event | (position, event) <- zip [0 :: Int ..] events])] ++ [finished]
    (_, produced) <- evalEither (attempt valid)
    map describe (drop 3 produced) === ["admitted [0]"]
    forM_ [0, 1, 2] $ \index -> assert (isLeft (attempt (altered index (Fixture.change "extra" Null))))
    assert (isLeft (attempt (records [object ["stage" .= String "load"]] ++ drop 1 valid)))
    assert (isLeft (attempt (take 3 valid ++ records [object ["stage" .= String "inference"]] ++ drop 4 valid)))
    assert (isLeft (attempt (take 2 valid ++ records [Serial.timer "inference"] ++ drop 2 valid)))
    assert (isLeft (attempt (take 2 valid ++ records [object ["stage" .= String "profile", "model" .= String "test-model", "revision" .= String "test-revision"]] ++ drop 2 valid)))
    assert (isLeft (attempt (take 4 valid ++ [unterminated (valid !! 4)])))
    assert (isLeft (attempt (take 4 valid ++ [finished])))
    assert (isLeft (attempt (take 5 valid ++ [Session.Ended (Transcript.Exited (ExitFailure 7) Transcript.Complete)])))
    assert (isLeft (attempt (take 5 valid ++ [valid !! 4, finished])))
    assert (isLeft (attempt (take 5 valid ++ [Session.Ended (Transcript.Exited ExitSuccess Transcript.Cut)])))
    assert (isLeft (attempt (take 5 valid ++ [Session.Ended (Transcript.Stopped Transcript.Complete)])))
  where
    unterminated (Session.Line raw) = Session.Fragment raw
    unterminated supplied = supplied

profiled :: PropertyT IO ()
profiled = do
    (first, _) <- serial
    batch <- Batched.setup [0, 1]
    let loading name = [object ["stage" .= String "loading"], object ["stage" .= String "profile", "model" .= String name, "revision" .= String "test-revision"], Serial.timer "load"]
        serialRun name = feed (Session.start Session.Serial) (Session.Dispatched (Session.Declaration [fst first] Nothing) : records (loading name) ++ drop 1 (session [first]) ++ [finished])
        batchedRun name = feed (Session.start Session.Batched) (Session.Dispatched (Session.Declaration (map fst batch) Nothing) : records (loading name ++ drop 1 (Batched.prefix batch) ++ Batched.suffix batch) ++ [finished])
    (_, serialProducts) <- evalEither (serialRun "test-model")
    map describe (drop 3 serialProducts) === ["admitted [0]"]
    (_, batchedProducts) <- evalEither (batchedRun "test-model")
    map describe (drop 3 batchedProducts) === ["admitted [0,1]"]
    assert (isLeft (serialRun "other-model"))
    assert (isLeft (batchedRun "other-model"))

referenced :: PropertyT IO ()
referenced = do
    (first, _) <- serial
    let (call, events) = first
        declared = replicate 64 'b'
        scored adapter = [if Fixture.field "stage" event == String "result" then Fixture.change "reference" (object ["adapter" .= adapter, "bits" .= [0xbf000000, 0xbe800000 :: Word32]]) event else event | event <- events]
        attempt reference changed = feed (Session.start Session.Serial) (Session.Dispatched (Session.Declaration [call] reference) : session [(call, changed)] ++ [finished])
    (_, produced) <- evalEither (attempt (Just declared) (scored declared))
    map describe (drop 3 produced) === ["admitted [0]"]
    assert (isLeft (attempt (Just declared) (scored (replicate 64 'd'))))
    assert (isLeft (attempt Nothing (scored declared)))
    assert (isLeft (attempt (Just declared) events))

delimited :: PropertyT IO ()
delimited = do
    (first, second) <- serial
    let inputs = session [first, second]
        opened = Session.step (Session.start Session.Serial) (Session.Dispatched (Session.Declaration [fst first, fst second] Nothing))
        replayed supplied = opened >>= \(started, _) -> feed started supplied
    (_, produced) <- evalEither (replayed (inputs ++ [Session.Delimited]))
    map describe (filter admittedProduct produced) === ["admitted [0,1]"]
    assert (isLeft (replayed (take (length inputs - 1) inputs ++ [Session.Delimited])))
    assert (isLeft (replayed (take 5 inputs ++ [Session.Delimited])))
    root <- workspace
    group <- F.prepare root 0 [0, 1]
    (hosted, _) <- evalEither (Session.step (Session.start Session.Resident) (Session.Hosted (Owner.start (Owner.Owner Owner.Inference F.owner)) (Session.Declaration (F.calls group) Nothing)))
    assert (isLeft (feed hosted (records (F.before group ++ F.after group) ++ [Session.Delimited])))
  where
    admittedProduct (Session.Admitted _) = True
    admittedProduct _ = False

resident :: PropertyT IO ()
resident = do
    root <- workspace
    first <- F.prepare root 0 [0, 1]
    second <- F.prepare root 1 [2, 3]
    let physical = Owner.start (Owner.Owner Owner.Inference F.owner)
        start = Session.start Session.Resident
    (dispatched, opening) <- evalEither (Session.step start (Session.Hosted physical (Session.Declaration (F.calls first) Nothing)))
    map describe opening === ["request [0,1]"]
    (granted, permitted) <- evalEither (feed dispatched (records (F.before first)))
    map describe permitted === ["send"]
    (releasing, released) <- evalEither (feed granted (records (F.after first)))
    map describe released === ["send"]
    (idle, acknowledged) <- evalEither (feed releasing (records [F.released first]))
    map describe acknowledged === ["owned", "admitted [0,1]"]
    following <- case [current | Session.Owned current <- acknowledged] of
        [current] -> pure current
        _ -> failure
    Owner.groups following === 1
    (interrupted, _) <- evalEither (Session.step idle (Session.Hosted following (Session.Declaration (F.calls second) Nothing)))
    assert (isLeft (feed interrupted (take 1 (records (F.before second)) ++ [Session.Ended (Transcript.Exited (ExitFailure 1) Transcript.Complete)])))

owned :: PropertyT IO ()
owned = do
    let selected = Owner.Owner Owner.Inference 0
        bound index = V.Binding (V.CallId index) (V.AttemptId index) (V.Instance index)
        model name = case object ["model" .= String name, "revision" .= String "test-revision"] of
            Object fields -> fields
            _ -> mempty
        acknowledged measurement = Lazy.toStrict (encode (object ["measurement" .= (measurement <> "\n" :: Text)]))
        cpu = "{\"stage\":\"released\",\"cpu_seconds\":0.1}"
        worker = "{\"stage\":\"released\",\"seconds\":0.1,\"peak_allocated\":1,\"peak_reserved\":1}"
        released index name measurement = Owner.Released [] [bound index] [model name] (acknowledged measurement)
        closing groups = Bytes.pack ("{\"format\":\"invar-resident-v1\",\"owner\":{\"role\":\"inference\",\"session\":0},\"groups\":" ++ show (groups :: Int) ++ ",\"stage\":\"closed\",\"measurement\":\"{\\\"stage\\\":\\\"closed\\\",\\\"cpu_seconds\\\":0.1}\\n\"}")
    first <- evalEither (Owner.release (Owner.start selected) (Owner.Released ["{\"stage\":\"load\",\"cpu_seconds\":0.25}"] [bound 0] [model "test-model"] (acknowledged cpu)))
    assert (isLeft (Owner.release first (released 0 "test-model" cpu)))
    assert (isLeft (Owner.release first (released 1 "other-model" cpu)))
    assert (isLeft (Owner.release first (released 1 "test-model" worker)))
    second <- evalEither (Owner.release first (released 1 "test-model" cpu))
    Owner.groups second === 2
    evalEither (Owner.close second (closing 2))
    assert (isLeft (Owner.close second (closing 1)))

evidenced :: PropertyT IO ()
evidenced = do
    (call, events) <- Fixture.setup
    (_, produced) <- evalEither (feed (Session.start Session.Single) (Session.Dispatched (Session.Declaration [call] Nothing) : session [(call, events)] ++ [finished]))
    case [trajectory | Session.Admitted [trajectory] <- produced] of
        [trajectory] -> do
            annotate (Bytes.unpack (Trajectory.evidence trajectory))
            decodeUtf8 (Trajectory.evidence trajectory) === expectedEvidence
            Trajectory.digest trajectory === expectedDigest
        _ -> failure

expectedEvidence :: Text
expectedEvidence = "{\"call\":{\"binding\":{\"attempt\":11,\"call\":7,\"instance\":13},\"load\":{\"binding\":{\"attempt\":11,\"call\":7,\"instance\":13},\"program\":\"(program 1 (sources ((semantic \\\"policy\\\") (record (\\\"artifact\\\" (sequence token)) (\\\"profile\\\" (sequence token))))) (signatures) (meanings) (sinks (\\\"load\\\" (\\\"policy-load/v1\\\" (record (\\\"artifact\\\" (sequence token)) (\\\"profile\\\" (sequence token))) (allowed (semantic \\\"policy\\\")) (recorded)))) (commands (emit \\\"load\\\" \\\"policy-load/v1\\\" (input (semantic \\\"policy\\\")))))\"},\"program\":\"(program 1 (sources ((semantic \\\"policy\\\") (record (\\\"artifact\\\" (sequence token)) (\\\"profile\\\" (sequence token)))) ((semantic \\\"request\\\") (record (\\\"artifact\\\" (sequence token)) (\\\"assembly\\\" (sequence token)) (\\\"base\\\" (sequence token)) (\\\"prompt\\\" (sequence token)) (\\\"temperature\\\" number) (\\\"tokenizer\\\" (sequence token)) (\\\"tokens\\\" token))) ((random \\\"sample\\\") number)) (signatures) (meanings) (sinks (\\\"infer\\\" (\\\"categorical-inference/v1\\\" (record (\\\"policy\\\" (record (\\\"artifact\\\" (sequence token)) (\\\"profile\\\" (sequence token)))) (\\\"request\\\" (record (\\\"artifact\\\" (sequence token)) (\\\"assembly\\\" (sequence token)) (\\\"base\\\" (sequence token)) (\\\"prompt\\\" (sequence token)) (\\\"temperature\\\" number) (\\\"tokenizer\\\" (sequence token)) (\\\"tokens\\\" token))) (\\\"seed\\\" number)) (allowed (semantic \\\"policy\\\") (semantic \\\"request\\\") (random \\\"sample\\\")) (recorded)))) (commands (emit \\\"infer\\\" \\\"categorical-inference/v1\\\" (fields (\\\"policy\\\" (input (semantic \\\"policy\\\"))) (\\\"request\\\" (input (semantic \\\"request\\\"))) (\\\"seed\\\" (random (random \\\"sample\\\")))))))\"},\"declared\":{\"adapter\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"assembly\":\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\",\"base\":\"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\",\"kind\":\"materialization\",\"tokenizer\":\"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\"},\"format\":\"invar-trajectory-evidence-v1\",\"observed\":{\"assembly\":\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\",\"base\":\"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\",\"behavior_bits\":[3204448256,3196059648],\"consumed\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"image\":{\"artifact\":\"3de2adea185acf25cea62e43447a9bbeb40ca178a3fdf5e92b979352fa5800d2\",\"profile\":\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\"},\"model\":\"test-model\",\"prompt_tokens\":[1],\"reference\":null,\"requested\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"response_tokens\":[2,3],\"revision\":\"test-revision\",\"text\":\"#### 12\",\"tokenizer\":\"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\",\"truncated\":true},\"request\":{\"prompt\":\"Compute the answer.\",\"seed\":17,\"temperature\":4605380978949069210,\"tokens\":2}}"

expectedDigest :: String
expectedDigest = "5de7dcba97c81d2c6dafd05090a69492df548f3920f1f9cd875424a29f41bacc"

canonical :: PropertyT IO ()
canonical = do
    let value =
            Canonical.Object
                ( Map.fromList
                    [ ("b", Canonical.Array [Canonical.Integer (-3), Canonical.Null, Canonical.Boolean True, Canonical.Boolean False])
                    , ("a", Canonical.Text "quote \" backslash \\ newline \n tab \t accent \233")
                    , ("\233", Canonical.Object (Map.fromList [("z", Canonical.Integer 0), ("A", Canonical.Text "")]))
                    , ("Z", Canonical.Integer 18446744073709551616)
                    ]
                )
    decodeUtf8 (Canonical.encode value) === "{\"Z\":18446744073709551616,\"a\":\"quote \\\" backslash \\\\ newline \\u000a tab \\u0009 accent \233\",\"b\":[-3,null,true,false],\"\233\":{\"A\":\"\",\"z\":0}}"

substitute :: ByteString -> ByteString -> ByteString -> ByteString
substitute old new raw = case Bytes.breakSubstring old raw of
    (before, after) | not (Bytes.null after) -> before <> new <> Bytes.drop (Bytes.length old) after
    _ -> raw
