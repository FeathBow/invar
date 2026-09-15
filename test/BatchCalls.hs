{-# LANGUAGE OverloadedStrings #-}

module BatchCalls (batchCalls, prepared, exchange, quote) where

import Calls qualified as Fixture
import Control.Monad (forM_)
import Data.Aeson (Value (..), eitherDecodeStrict, object, (.=))
import Data.ByteString.Char8 qualified as Bytes
import Hedgehog
import Invar.Infer qualified as I
import Invar.Infer.Invocation qualified as C
import Invar.Spec.Invocation qualified as V
import Invar.Worker qualified as W
import Numeric.Natural (Natural)
import Store (workspace)
import System.Exit (ExitCode (ExitFailure))
import System.FilePath ((</>))

batchCalls :: Group
batchCalls =
    Group
        "Persistent batch protocol"
        [ ("one process performs distinct request and permission exchanges", once completed)
        , ("worker configuration is passed intact to the persistent process", once configured)
        , ("invalid consumption or result prevents subsequent dispatch", once rejected)
        , ("completion requires clean exit and no trailing output", once terminal)
        , ("adapter replacement requires the exact previous unload", once lifetime)
        ]
  where
    once = withTests 1 . property

setup :: PropertyT IO [(C.Call, [Value])]
setup = do
    (_, events) <- Fixture.setup
    planned <- evalEither (I.prepare Fixture.request)
    traverse (\index -> prepared planned events index (if index > 0 then Just (index - 1) else Nothing)) [0 .. 2]

prepared :: I.Plan -> [Value] -> Natural -> Maybe Natural -> PropertyT IO (C.Call, [Value])
prepared planned events index previous = do
    let bound = V.Binding (V.CallId index) (V.AttemptId index) (V.Instance index)
    call <- evalEither (C.prepare bound planned)
    envelope <- evalEither (eitherDecodeStrict (C.batchInput call))
    let binding = Fixture.field "binding" envelope
        loading = Fixture.field "load" envelope
        rebound = map (Fixture.change "binding" binding) events
        updated = map (Fixture.change "load" loading) (take 2 rebound) ++ drop 2 rebound
        unloaded earlier = Fixture.change "stage" (String "unloaded_adapter") (Fixture.change "binding" (object ["call" .= earlier, "attempt" .= earlier, "instance" .= earlier]) loading)
    pure (call, maybe [] (pure . unloaded) previous ++ updated)

quote :: String -> String
quote text = "'" ++ concatMap (\character -> if character == '\'' then "'\\''" else [character]) text ++ "'"

exchange :: FilePath -> (C.Call, [Value]) -> String
exchange root (call, events) =
    unlines
        [ "IFS= read -r request || exit 21"
        , "printf '%s\\n' received >> " ++ quote (root </> "received")
        , "test \"$request\" = " ++ quote (Bytes.unpack (C.batchInput call)) ++ " || exit 22"
        , emit (Fixture.reviewPrefix events)
        , "IFS= read -r permission || exit 23"
        , "test \"$permission\" = " ++ quote (Bytes.unpack (Fixture.permissionInput call)) ++ " || exit 24"
        , emit (drop (length (Fixture.reviewPrefix events)) events)
        ]
  where
    emit [] = "exit 0"
    emit values = "printf '%s\\n' " ++ unwords (map (quote . Bytes.unpack) (Bytes.lines (Fixture.wire values)))

run :: [(C.Call, [Value])] -> String -> PropertyT IO (Either W.Failure [W.Execution], Int)
run = runWith Nothing

runWith :: Maybe FilePath -> [(C.Call, [Value])] -> String -> PropertyT IO (Either W.Failure [W.Execution], Int)
runWith configuration requests ending = do
    root <- workspace
    let script = root </> "batch.sh"
        configuredArgument = maybe "" (\path -> "test \"$3\" = " ++ quote ("--config=" ++ path) ++ " || exit 20\n") configuration
        body = configuredArgument ++ "printf '%s\\n' launched >> " ++ quote (root </> "launched") ++ "\n" ++ concatMap (exchange root) requests ++ ending
    evalIO (writeFile script body)
    returned <- evalIO (W.runBatch (W.Worker "/bin/sh" script root "unused" [] configuration) (map fst requests))
    launches <- evalIO (readFile (root </> "launched"))
    launches === "launched\n"
    received <- evalIO (readFile (root </> "received"))
    pure (returned, length (lines received))

configured :: PropertyT IO ()
configured = do
    requests <- setup
    (outcome, count) <- runWith (Just "native configuration.json") requests "IFS= read -r extra && exit 25\nexit 0\n"
    values <- evalEither outcome
    count === length requests
    length values === length requests

completed :: PropertyT IO ()
completed = do
    requests <- setup
    (outcome, count) <- run requests "IFS= read -r extra && exit 25\nexit 0\n"
    values <- evalEither outcome
    count === length requests
    map (V.completedBinding . W.completion) values === [V.Binding (V.CallId index) (V.AttemptId index) (V.Instance index) | index <- [0 .. 2]]
    forM_ requests $ \(call, _) -> do
        envelope <- evalEither (eitherDecodeStrict (C.batchInput call))
        case envelope of
            Object _ -> success
            _ -> failure

rejected :: PropertyT IO ()
rejected = do
    requests <- setup
    let corrupt [loaded, consumed, result] =
            [ [loaded, Fixture.change "program" (String "other") consumed, result]
            , [loaded, consumed, Fixture.change "text" (Number 1) result]
            , [loaded, consumed, consumed, result]
            , [loaded, consumed]
            ]
        corrupt _ = error "Expected three protocol stages"
    case requests of
        (call, events) : remaining -> forM_ (corrupt events) $ \changed -> do
            (outcome, count) <- run ((call, changed) : remaining) "exit 0\n"
            count === 1
            case outcome of
                Left (W.InvalidOutput _) -> success
                Left (W.ProtocolFailure _) -> success
                _ -> failure
        _ -> failure

terminal :: PropertyT IO ()
terminal = do
    requests <- setup
    forM_ ["exit 7\n", "printf '%s\\n' trailing\nexit 0\n"] $ \ending -> do
        (outcome, count) <- run requests ending
        count === length requests
        case outcome of
            Left (W.WorkerExit (ExitFailure 7)) -> success
            Left (W.ProtocolFailure _) -> success
            _ -> failure

lifetime :: PropertyT IO ()
lifetime = do
    requests <- setup
    case requests of
        first : (call, unloaded : events) : remaining -> do
            let wrongBinding = Fixture.change "binding" (object ["call" .= Number 99, "attempt" .= Number 99, "instance" .= Number 99]) unloaded
                wrongProgram = Fixture.change "program" (String "wrong") unloaded
            forM_ [events, unloaded : unloaded : events, wrongBinding : events, wrongProgram : events] $ \changed -> do
                (outcome, count) <- run (first : (call, changed) : remaining) "exit 0\n"
                count === 2
                case outcome of
                    Left (W.InvalidOutput _) -> success
                    _ -> failure
        _ -> failure
