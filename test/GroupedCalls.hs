{-# LANGUAGE OverloadedStrings #-}

module GroupedCalls (groupedCalls) where

import BatchCalls qualified as Batch
import Calls qualified as Fixture
import Control.Monad (forM_)
import Data.Aeson (Value (..), eitherDecodeStrict, object, (.=))
import Data.ByteString (ByteString)
import Hedgehog
import Invar.Infer qualified as Infer
import Invar.Infer.Invocation qualified as Call
import Invar.Spec.Invocation qualified as Invocation
import Invar.Spec.Load qualified as Load

groupedCalls :: Group
groupedCalls =
    Group
        "Finite activation groups"
        [ ("all batch activations stay live while distinct permits complete", once completed)
        , ("one invalid member cannot authorize the finite group", once rejected)
        , ("batch calls attempts and instances must be independently distinct", once identities)
        , ("active serial loads cannot be retired to make room for a batch", once active)
        ]
  where
    once = withTests 1 . property

setup :: PropertyT IO [(Call.Call, [Value])]
setup = do
    (_, events) <- Fixture.setup
    planned <- evalEither (Infer.prepare Fixture.request)
    traverse (\index -> Batch.prepared planned events index Nothing) [0 .. 2]

ready :: [(Call.Call, [Value])] -> [(Call.Call, ByteString)]
ready = map (\(call, events) -> (call, Fixture.wire (Fixture.reviewPrefix events)))

completed :: PropertyT IO ()
completed = do
    requests <- setup
    (registry, permits) <- evalEither (Call.authorizeBatch Load.empty (ready requests))
    Load.active registry === map (Invocation.boundInstance . Call.binding . fst) requests
    forM_ (zip requests permits) $ \((call, events), permit) -> do
        Call.permission permit === Fixture.permissionInput call
        (finished, _) <- evalEither (Call.observe permit (Fixture.wire events))
        Invocation.completedBinding finished === Call.binding call
        Invocation.completedBinding (Load.report (Call.loadFact permit)) === Call.binding call
    Load.active registry === map (Invocation.boundInstance . Call.binding . fst) requests
    let closed = Load.close registry
    Load.active closed === []
    isLeft (Call.authorizeBatch closed (ready requests))

rejected :: PropertyT IO ()
rejected = do
    requests <- setup
    isLeft (Call.authorizeBatch Load.empty [])
    case reverse requests of
        (call, [loaded, consumed, result]) : previous -> do
            let mutate fields key value = Fixture.change key value fields
                image = object ["artifact" .= String "wrong", "profile" .= String "wrong"]
                malformed =
                    [ [loaded, mutate consumed "program" (String "different"), result]
                    , [mutate loaded "image" image, consumed, result]
                    , [loaded, mutate consumed "load" Null, result]
                    , [loaded, consumed, consumed, result]
                    , [loaded, result]
                    , [mutate loaded "stage" (String "unloaded_adapter"), loaded, consumed, result]
                    ]
            forM_ malformed $ \events -> do
                let submitted = takeWhile (\event -> Fixture.field "stage" event /= String "result") events
                isLeft (Call.authorizeBatch Load.empty (ready (reverse previous) ++ [(call, Fixture.wire submitted)]))
        _ -> failure

identities :: PropertyT IO ()
identities = do
    requests <- setup
    planned <- evalEither (Infer.prepare Fixture.request)
    case requests of
        (first, events) : remaining -> do
            let bound = Call.binding first
                changed = Invocation.Binding (Invocation.CallId 99) (Invocation.AttemptId 99) (Invocation.Instance 99)
                aliases = [bound, changed {Invocation.boundCall = Invocation.boundCall bound}, changed {Invocation.boundAttempt = Invocation.boundAttempt bound}, changed {Invocation.boundInstance = Invocation.boundInstance bound}]
            forM_ aliases $ \alias -> do
                call <- evalEither (Call.prepare alias planned)
                output <- rebound call events
                isLeft (Call.authorizeBatch Load.empty (ready ((first, events) : (call, output) : remaining)))
        _ -> failure

rebound :: Call.Call -> [Value] -> PropertyT IO [Value]
rebound call events = do
    envelope <- evalEither (eitherDecodeStrict (Call.batchInput call))
    let bound = Fixture.field "binding" envelope
        loading = Fixture.field "load" envelope
        program = Fixture.field "program" envelope
    case map (Fixture.change "binding" bound) events of
        [loaded, consumed, result] -> pure [Fixture.change "load" loading loaded, Fixture.change "program" program (Fixture.change "load" loading consumed), result]
        _ -> failure >> pure []

active :: PropertyT IO ()
active = do
    requests <- setup
    case requests of
        (first, events) : remaining -> do
            (registry, _) <- evalEither (Call.authorize Load.empty first (Fixture.wire (Fixture.reviewPrefix events)))
            isLeft (Call.authorizeBatch registry (ready remaining))
            Load.active registry === [Invocation.boundInstance (Call.binding first)]
            forM_ remaining $ \(call, output) -> isLeft (Call.authorize registry call (Fixture.wire (Fixture.reviewPrefix output)))
        _ -> failure

isLeft :: Either problem value -> PropertyT IO ()
isLeft (Left _) = success
isLeft (Right _) = failure
