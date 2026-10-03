{-# LANGUAGE OverloadedStrings #-}

module BatchedProtocol (batchedProtocol, setup, prefix, suffix, input, permission) where

import BatchCalls qualified as Serial
import Calls qualified as Fixture
import Control.Monad (forM_)
import Data.Aeson (Value (..), encode, object, toJSON, (.=))
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.IORef (modifyIORef', newIORef, readIORef, writeIORef)
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8, encodeUtf8)
import Hedgehog
import Invar.Cohort qualified as Cohort
import Invar.Infer qualified as Infer
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Result qualified as Result
import Invar.Reward qualified as Reward
import Invar.Rollout qualified as Rollout
import Invar.Spec.Invocation qualified as Invocation
import Invar.Transcript qualified as Transcript
import Invar.Worker qualified as Worker
import Numeric.Natural (Natural)
import Store (workspace)
import System.Directory (doesFileExist)
import System.Exit (ExitCode (ExitFailure))
import System.FilePath ((</>))
import System.IO.Error (tryIOError)

batchedProtocol :: Group
batchedProtocol =
    Group
        "Finite batched process protocol"
        [ ("one batch frame grants all distinct calls and retains original observations", once completed)
        , ("invalid or incomplete group readiness grants no permission", once readiness)
        , ("batch completion rejects a wrong missing extra or reordered result", once results)
        , ("batch completion preserves the original probability zero sign", once zeroSign)
        , ("batch completion requires clean exit and no trailing output, and records the trailing output", once terminal)
        , ("batched rollout retains device partition and logical delivery", once rollout)
        , ("a declared reference reaches both finite worker protocols", once declared)
        , ("a worker that exits before reading its input reports its exit status", once unread)
        , ("a transcript learns whether its process exited, was stopped by the core or never started", once ended)
        ]
  where
    once = withTests 1 . property

format :: Text
format = "invar-inference-batch-v1"

setup :: [Natural] -> PropertyT IO [(Call.Call, [Value])]
setup indices = do
    (_, events) <- Fixture.setup
    planned <- evalEither (Infer.prepare Fixture.request)
    traverse (\index -> Serial.prepared planned events index Nothing) indices

frame :: Text -> [ByteString] -> Value
frame stage calls = object ["stage" .= stage, "format" .= format, "calls" .= map decodeUtf8 calls]

duration :: Text -> Value
duration stage = object ["stage" .= stage, "cpu_seconds" .= (0.25 :: Double)]

prefix :: [(Call.Call, [Value])] -> [Value]
prefix requests = [duration "load", frame "consumed" (map (Fixture.wire . Fixture.reviewPrefix . snd) requests)]

suffix :: [(Call.Call, [Value])] -> [Value]
suffix requests = [duration "inference", frame "result" (map (Fixture.wire . pure . last . snd) requests)]

input :: FilePath -> [Call.Call] -> ByteString
input adapter = inputWith adapter Null

inputWith :: FilePath -> Value -> [Call.Call] -> ByteString
inputWith adapter reference calls = Lazy.toStrict (encode (object ["format" .= format, "adapter" .= adapter, "reference" .= reference, "calls" .= map (decodeUtf8 . Call.batchInput) calls]))

permission :: [Call.Call] -> ByteString
permission calls = Lazy.toStrict (encode (object ["format" .= format, "permissions" .= map (decodeUtf8 . Fixture.permissionInput) calls]))

script :: FilePath -> [Call.Call] -> ([Value], [Value], String) -> String
script root calls (before, after, ending) =
    unlines
        [ "IFS= read -r request || exit 21"
        , "test \"$request\" = " ++ quote (Bytes.unpack (input "adapter path" calls)) ++ " || exit 22"
        , emit before
        , "IFS= read -r permission || exit 23"
        , "test \"$permission\" = " ++ quote (Bytes.unpack (permission calls)) ++ " || exit 24"
        , "printf '%s\\n' approved > " ++ quote (root </> "approved")
        , emit after
        , ending
        ]
  where
    quote = Serial.quote
    emit values = "printf '%s\\n' " ++ unwords (map (quote . Bytes.unpack) (Bytes.lines (Fixture.wire values)))

clean :: String
clean = "IFS= read -r extra && exit 25\nexit 0"

run :: [(Call.Call, [Value])] -> ([Value], [Value], String) -> PropertyT IO (Either Worker.Failure [Worker.Execution], Bool, ByteString)
run requests observations = do
    root <- workspace
    let path = root </> "finite.sh"
        configuration = "native configuration.json"
        argument = "test \"$2\" = " ++ Serial.quote ("--config=" ++ configuration) ++ " || exit 20\n"
        worker = Worker.Worker "/bin/sh" path root "adapter path" [] (Just configuration)
    evalIO (writeFile path (argument ++ script root (map fst requests) observations))
    output <- evalIO (newIORef [])
    returned <- evalIO (Worker.runBatchedSession worker Nothing (Transcript.echoing (\line -> modifyIORef' output (line :))) (map fst requests))
    approved <- evalIO (doesFileExist (root </> "approved"))
    emitted <- evalIO (Bytes.unlines . reverse <$> readIORef output)
    pure (returned, approved, emitted)

completed :: PropertyT IO ()
completed = do
    requests <- setup [2, 0, 1]
    (outcome, approved, emitted) <- run requests (prefix requests, suffix requests, clean)
    approved === True
    emitted === Fixture.wire (prefix requests ++ suffix requests)
    values <- evalEither outcome
    map (Invocation.completedBinding . Worker.completion) values === map (Call.binding . fst) requests
    forM_ (zip values requests) $ \(executed, (_, events)) -> do
        Invocation.completedOutput (Worker.completion executed) === Lazy.toStrict (encode (last events))
        Result.behaviorBits (Worker.report executed) === [0xbf000000, 0xbe800000]

readiness :: PropertyT IO ()
readiness = do
    requests <- setup [2, 0, 1]
    let members = map (Fixture.wire . Fixture.reviewPrefix . snd) requests
        invalid =
            [ []
            , take 2 members
            , members ++ take 1 members
            , reverse members
            , take 1 members ++ take 1 members ++ drop 2 members
            , init members ++ [Fixture.wire (take 1 (snd (last requests)))]
            , init members ++ [Fixture.wire (snd (last requests))]
            , init members ++ [Fixture.wire (map (Fixture.change "program" (String "wrong")) (Fixture.reviewPrefix (snd (last requests))))]
            ]
        outer = [[duration "load", frame "consumed" values] | values <- invalid]
        malformed = [[frame "consumed" members], [duration "load", duration "load", frame "consumed" members], [duration "load", frame "result" members]]
    forM_ (outer ++ malformed) $ \before -> do
        (outcome, approved, _) <- run requests (before, suffix requests, clean)
        approved === False
        rejected outcome

results :: PropertyT IO ()
results = do
    requests <- setup [2, 0, 1]
    let members = map (Fixture.wire . pure . last . snd) requests
        invalid = [[], take 2 members, members ++ take 1 members, reverse members, take 1 members ++ take 1 members ++ drop 2 members]
        wrong = Fixture.wire [Fixture.change "text" (Number 1) (last (snd (last requests)))]
        outer = [[duration "inference", frame "result" values] | values <- invalid ++ [init members ++ [wrong]]]
        malformed = [[frame "result" members], [duration "inference", duration "inference", frame "result" members], prefix requests, [duration "inference"]]
    forM_ (outer ++ malformed) $ \after -> do
        (outcome, approved, _) <- run requests (prefix requests, after, "exit 0")
        approved === True
        rejected outcome

zeroSign :: PropertyT IO ()
zeroSign = do
    requests@[(_, events)] <- setup [0]
    let result = last events
        bits = Fixture.change "behavior_bits" (toJSON [0x80000000, 0xbe800000 :: Integer]) result
        changed = Text.replace "[-0.5,-0.25]" "[-0.0,-0.25]" (decodeUtf8 (Fixture.wire [bits]))
    forM_ [True, False] $ \negative -> do
        let raw = encodeUtf8 (if negative then changed else Text.replace "[-0.0," "[0.0," changed)
        (outcome, approved, _) <- run requests (prefix requests, [duration "inference", frame "result" [raw]], clean)
        approved === True
        if negative
            then do
                values <- evalEither outcome
                map (Result.behaviorBits . Worker.report) values === [[0x80000000, 0xbe800000]]
            else rejected outcome
terminal :: PropertyT IO ()
terminal = do
    requests <- setup [0, 1]
    forM_ ["exit 7", "printf '%s\\n' trailing\nexit 0"] $ \ending -> do
        (outcome, approved, emitted) <- run requests (prefix requests, suffix requests, ending)
        approved === True
        case outcome of
            Left (Worker.WorkerExit (ExitFailure 7)) -> emitted === Fixture.wire (prefix requests ++ suffix requests)
            Left (Worker.ProtocolFailure _) -> emitted === Fixture.wire (prefix requests ++ suffix requests) <> "trailing\n"
            _ -> failure

rollout :: PropertyT IO ()
rollout = forM_ [1, 2] $ \count -> do
    root <- workspace
    planned <- evalEither (Infer.prepare Fixture.request)
    expected <- evalEither (Reward.decimal "#### 12")
    let order = [2, 0, 1]
        groups = [[index | (position, index) <- zip [0 :: Int ..] order, position `mod` count == slot] | slot <- [0 .. count - 1]]
    requests <- traverse setup groups
    let path = root </> "rollout.sh"
        branch slot calls = show slot ++ ")\n" ++ script root (map fst calls) (prefix calls, suffix calls, clean) ++ ";;"
        body = unlines (["case \"$INVAR_TEST_SESSION\" in"] ++ zipWith branch [0 :: Int ..] requests ++ ["*) exit 31;;", "esac"])
        worker = Worker.Worker "/bin/sh" path root "adapter path" [] Nothing
        tasks = [Cohort.Task ("member" ++ show index) "group" planned expected | index <- [0 :: Int .. 2]]
        options = Rollout.Options worker Rollout.Batched [[("INVAR_TEST_SESSION", show slot)] | slot <- [0 .. count - 1]] (Cohort.Definition (Infer.artifact Fixture.request) tasks) order [1, 2, 0] Nothing
        project batch = (map Rollout.name (Rollout.samples batch), Rollout.delivered batch)
    evalIO (writeFile path body)
    returned <- evalIO (Rollout.withDriver (\driver -> fmap project <$> Rollout.run driver options)) >>= evalEither
    fst returned === ["member0", "member1", "member2"]
    snd returned === [Invocation.Binding (Invocation.CallId index) (Invocation.AttemptId index) (Invocation.Instance index) | index <- [1, 2, 0]]

rejected :: Either Worker.Failure value -> PropertyT IO ()
rejected (Left (Worker.InvalidOutput _)) = success
rejected (Left (Worker.ProtocolFailure _)) = success
rejected (Left problem) = annotateShow problem >> failure
rejected (Right _) = failure

declared :: PropertyT IO ()
declared = do
    requests <- setup [2, 0, 1]
    root <- workspace
    let selected = Worker.Reference "reference adapter" (replicate 64 'c')
        expected = object ["adapter" .= String "reference adapter", "digest" .= String (Text.replicate 64 "c")]
        batched = root </> "batched.sh"
        serial = root </> "serial.sh"
        arguments = root </> "arguments"
    evalIO (writeFile batched ("IFS= read -r request || exit 21\ntest \"$request\" = " ++ Serial.quote (Bytes.unpack (inputWith "adapter path" expected (map fst requests))) ++ " || exit 22\nexit 23\n"))
    batchedOutcome <- evalIO (Worker.runBatchedSession (Worker.Worker "/bin/sh" batched root "adapter path" [] Nothing) (Just selected) (Transcript.echoing (const (pure ()))) (map fst requests))
    case batchedOutcome of
        Left problem -> problem === Worker.WorkerExit (ExitFailure 23)
        Right _ -> failure
    evalIO (writeFile serial ("printf '%s\\n' \"$@\" > " ++ Serial.quote arguments ++ "\nexit 23\n"))
    _ <- evalIO (Worker.runSession (Worker.Worker "/bin/sh" serial root "adapter path" [] Nothing) (Just selected) (Transcript.echoing (const (pure ()))) (map fst requests))
    received <- evalIO (lines <$> readFile arguments)
    drop 2 received === ["--reference=reference adapter", "--reference-digest=" ++ replicate 64 'c']

unread :: PropertyT IO ()
unread = do
    requests <- setup [0 .. 511]
    root <- workspace
    let exiting = root </> "unread.sh"
    evalIO (writeFile exiting "exit 23\n")
    annotateShow (Bytes.length (inputWith "adapter path" Null (map fst requests)))
    outcome <- evalIO (Worker.runBatchedSession (Worker.Worker "/bin/sh" exiting root "adapter path" [] Nothing) Nothing (Transcript.echoing (const (pure ()))) (map fst requests))
    case outcome of
        Left problem -> problem === Worker.WorkerExit (ExitFailure 23)
        Right _ -> failure

ended :: PropertyT IO ()
ended = do
    requests <- setup [2, 0, 1]
    root <- workspace
    let exiting = root </> "exiting.sh"
        refused = root </> "refused.sh"
        observe executable path = do
            outcome <- newIORef Nothing
            _ <- tryIOError (Worker.runBatchedSession (Worker.Worker executable path root "adapter path" [] Nothing) Nothing (Transcript.Transcript (const (pure ())) (writeIORef outcome . Just)) (map fst requests))
            readIORef outcome
    evalIO (writeFile exiting "exit 23\n")
    evalIO (writeFile refused (script root (map fst requests) ([duration "load", frame "consumed" []], [], clean)))
    evalIO (observe "/bin/sh" exiting) >>= (=== Just (Transcript.Exited (ExitFailure 23)))
    evalIO (observe "/bin/sh" refused) >>= (=== Just Transcript.Stopped)
    missing <- evalIO (observe (root </> "missing") exiting)
    case missing of
        Just (Transcript.Unlaunched _) -> success
        _ -> annotateShow missing >> failure
