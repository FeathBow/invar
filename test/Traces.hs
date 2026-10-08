{-# LANGUAGE OverloadedStrings #-}

module Traces (traces, built, admit) where

import Calls qualified
import Control.Monad (forM_)
import Data.Aeson (Value (..), encode, object, toJSON, (.=))
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Either (isLeft)
import Data.IORef (atomicModifyIORef', newIORef, readIORef)
import Data.List (isInfixOf)
import Data.Map.Strict qualified as Map
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8)
import Hedgehog
import Invar.Cohort qualified as C
import Invar.History.Cohort qualified as Cohort
import Invar.History.Generation qualified as Generation
import Invar.History.Trace qualified as Trace
import Invar.Infer qualified as I
import Invar.Learn qualified as L
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as R
import Invar.Spec.Invocation qualified as V
import Invar.Transcript qualified as Transcript
import Invar.Workload qualified as Workload
import LearnerFixture qualified as F
import Numeric.Natural (Natural)
import Sessions qualified
import Store (workspace)
import System.FilePath ((</>))

traces :: Group
traces =
    Group
        "Training history replay"
        [ ("a history replays each finite session, the learner and its publication under the declared policy", once admitted)
        , ("a generation is replayed under its declared policy description and reference, never under the worker's report", once declared)
        , ("a resident learner release is admitted by its acknowledgement's owner, loads and result digest", once released)
        , ("a session cut before its final result, or output after the history, is refused", once incomplete)
        ]
  where
    once = withTests 1 . property

data History = History {run :: Trace.Run, initial :: Policy.Description, document :: Workload.Document, inference :: [ByteString], learning :: [Value], closing :: [Value], published :: Value, finished :: Value}

members :: Natural
members = 5

assembled :: History -> ByteString
assembled history = Bytes.unlines (inference history ++ map line (learning history ++ [published history, finished history] ++ closing history))

line :: Value -> ByteString
line = Lazy.toStrict . encode

admit :: History -> Either String Trace.Checked
admit history = Trace.admit (run history) (initial history) (document history) (assembled history)

built :: Int -> Bool -> PropertyT IO History
built count resident = do
    root <- workspace
    chosen <- Sessions.options root count
    let settings = F.configured
        definition = R.definition chosen
    description <- evalEither (Policy.describe ("test-model", "test-revision") (L.policy settings, L.tokenizer settings, L.behaviorBase settings, L.behaviorAssembly settings))
    bound <- evalEither (traverse (\task -> either (Left . show) (\planned -> Right task {C.plan = planned}) (I.bindPolicy description (C.plan task))) (C.tasks definition))
    captured <- evalIO (newIORef Map.empty)
    let opened slot = pure (Transcript.Transcript (\recorded -> atomicModifyIORef' captured (\held -> (Map.insertWith (flip (++)) slot [recorded] held, ()))) (const (pure ())) (const (pure ())))
        update = V.ordinal members
    produced <- evalIO $ R.withRecordedDriver R.Serial (R.worker chosen, R.sessions chosen) opened $ \driver -> do
        batch <- R.run driver chosen {R.definition = definition {C.tasks = bound}} >>= F.require
        planned <- F.require (L.prepare settings batch)
        exchange <- F.prepare root (0, update) planned
        pure (F.before exchange, F.steps exchange, F.after exchange, F.released exchange)
    (before, steps, after, acknowledged) <- evalEither produced
    sessions <- evalIO (readIORef captured)
    result <- case reverse after of
        value : _ -> pure value
        [] -> failure
    let task index = object ["name" .= ("member" ++ show index), "group" .= String "group", "prompt" .= I.prompt Calls.request, "tokens" .= I.tokens Calls.request, "temperature" .= I.temperature Calls.request, "seed" .= I.seed Calls.request, "answer" .= String "#### 12"]
        workload = toJSON [object ["tasks" .= map task [0 .. members - 1], "order" .= R.order chosen, "delivery" .= R.delivery chosen]]
        output = root </> "train"
        bindingValue index = object ["call" .= index, "attempt" .= index, "instance" .= index]
        prefix = if resident then before else filter ((/= Just "activation") . F.stage) before
        closed = object ["stage" .= String "closed", "format" .= String "invar-resident-v1", "owner" .= F.owner, "groups" .= (1 :: Int), "measurement" .= decodeUtf8 (F.wire [F.timer "closed"])]
    decoded <- evalEither (Workload.decode (line workload))
    pure
        History
            { run = Trace.Run settings (fromIntegral count) output "rename" 0 Trace.Finite (if resident then Trace.Resident else Trace.Finite)
            , initial = description
            , document = decoded
            , inference = concat (Map.elems sessions)
            , learning = prefix ++ steps ++ after ++ [acknowledged | resident]
            , closing = [closed | resident]
            , published = object ["phase" .= String "published", "checkpoint" .= (output </> "generation1"), "policy" .= Calls.field "adapter" result, "learner" .= Calls.field "learner" result, "publication" .= String "rename", "binding" .= bindingValue members, "delivery" .= map bindingValue (R.delivery chosen)]
            , finished = object ["phase" .= String "cycle", "index" .= (0 :: Int), "sessions" .= count, "seconds" .= (0.5 :: Double)]
            }

admitted :: PropertyT IO ()
admitted = forM_ [(1, False), (2, False), (1, True)] $ \(count, resident) -> do
    history <- built count resident
    checked <- evalEither (admit history)
    case Trace.generations checked of
        [generation] -> length (Cohort.inferences (Generation.cohort generation)) === fromIntegral members
        _ -> failure
    toJSON (Policy.adapter (Trace.finalPolicy checked)) === Calls.field "policy" (published history)

declared :: PropertyT IO ()
declared = do
    history <- built 1 False
    let settings = Trace.settings (run history)
    other <- evalEither (Policy.describe ("other-model", "test-revision") (Policy.bindings (initial history)))
    assert (isLeft (admit history {initial = other}))
    case admit history {run = (run history) {Trace.settings = settings {L.reference = replicate 64 'd'}}} of
        Left problem -> assert ("Reference scores differ from the declared reference" `isInfixOf` problem)
        Right _ -> failure

released :: PropertyT IO ()
released = do
    history <- built 1 True
    _ <- evalEither (admit history)
    let altered = [if F.stage record == Just "released" then Calls.change "result_sha256" (String (Text.replicate 64 "0")) record else record | record <- learning history]
    assert (altered /= learning history)
    assert (isLeft (admit history {learning = altered}))

incomplete :: PropertyT IO ()
incomplete = do
    history <- built 2 False
    _ <- evalEither (admit history)
    assert (isLeft (admit history {inference = take (length (inference history) - 1) (inference history)}))
    assert (isLeft (admit history {closing = closing history ++ [object ["stage" .= String "result"]]}))
