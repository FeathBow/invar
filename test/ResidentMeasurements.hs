{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

module ResidentMeasurements (residentMeasurements) where

import Calls (change, field, request, wire)
import Control.Monad (forM_)
import Data.Aeson (Value (..), object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.List (nub)
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8)
import Hedgehog
import Invar.Evaluation qualified as Evaluation
import Invar.Infer qualified as Infer
import Invar.Measurement.Inference qualified as Measurement
import ResidentObservations qualified as Fixture
import Updates (alter)
import Workloads (array)

residentMeasurements :: Group
residentMeasurements =
    Group
        "Physical resident inference costs"
        [ ("initial load activation shared inference release and close are counted exactly once", once complete)
        , ("later cohorts use the original physical profile", once profiles)
        , ("unused owners retain real closing costs and strict clocks", once closing)
        ]
  where
    once = withTests 1 . property

run :: Evaluation.Run
run = Evaluation.Run (Infer.artifact request) 0

complete :: PropertyT IO ()
complete = forM_ [(1, 8.875), (2, 9), (3, 9.125)] $ \(owners, critical :: Double) -> do
    (tasks, events) <- Fixture.fixture owners
    observed <- evalEither (Measurement.admit tasks run (wire events))
    let reported = Measurement.describe observed
        active = min 2 owners
        loads = array (field "loads" reported)
        measured = array (field "measurements" reported)
    field "worker_mode" reported === String "resident"
    field "critical_path_seconds" reported === toJSON critical
    field "sessions_per_cohort" reported === toJSON (replicate 3 owners)
    field "active_sessions_per_cohort" reported === toJSON (replicate 3 active)
    field "model_loads_per_cohort" reported === toJSON [active, 0, 0]
    length loads === active
    length measured === active * 3
    length (array (field "closed" reported)) === owners
    forM_ loads $ \loaded -> do
        field "load" loaded === object ["cpu_seconds" .= Number 1]
        field "calls" loaded === toJSON (6 `div` active)
        field "requests_per_execution" loaded === toJSON (replicate 3 (2 `div` active))
    forM_ measured $ \measurement -> do
        field "inference" measurement === object ["cpu_seconds" .= Number 2]
        forM_ (array (field "calls" measurement)) $ \case
            Object fields -> assert (not (Fields.member "inference" fields))
            _ -> failure

profiles :: PropertyT IO ()
profiles = do
    (tasks, events) <- Fixture.fixture 1
    observed <- evalEither (Measurement.admit tasks run (wire events))
    let measured = array (field "measurements" (Measurement.describe observed))
        identities = [field "profile_sha256" sample | entry <- measured, sample <- array (field "calls" entry)]
    length identities === 6
    length (nub identities) === 1
    unreported <- Fixture.reseal [event | event <- events, not (stage "profile" event || stage "loading" event)]
    _ <- evalEither (Evaluation.admit tasks run (wire unreported))
    rejected (Measurement.admit tasks run (wire unreported))

closing :: PropertyT IO ()
closing = do
    (tasks, events) <- Fixture.fixture 3
    index <- case [position | (position, value) <- zip [0 ..] events, stage "closed" value] of
        first : _ -> pure first
        [] -> failure
    let changed timer = alter index (change "measurement" (String (decodeUtf8 (wire [timer])))) events
        expensive = object ["stage" .= String "closed", "cpu_seconds" .= (100.125 :: Double)]
        otherClock = object ["stage" .= String "closed", "seconds" .= (0.125 :: Double), "peak_allocated" .= Number 0, "peak_reserved" .= Number 0]
    observed <- evalEither (Measurement.admit tasks run (wire (changed expensive)))
    field "critical_path_seconds" (Measurement.describe observed) === toJSON (109.125 :: Double)
    _ <- evalEither (Evaluation.admit tasks run (wire (changed otherClock)))
    rejected (Measurement.admit tasks run (wire (changed otherClock)))

stage :: Text -> Value -> Bool
stage expected (Object fields) = Fields.lookup "stage" fields == Just (String expected)
stage _ _ = False

rejected :: Either String value -> PropertyT IO ()
rejected (Left _) = success
rejected (Right _) = failure
