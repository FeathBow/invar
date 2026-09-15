{-# LANGUAGE OverloadedStrings #-}

module Invar.Measurement.Run (Run (..), Schedule (..), Completion (..), completionFields, samples, measurements, loadDurations, costs, overhead, cohorts, counts, loadCounts, layout, residentOwners, clock, paired, describe, measurementFields, measurementEncoding, lifetimeFields) where

import Control.Monad (unless, when)
import Data.Aeson (Value (..), object, toJSON, (.=))
import Data.Aeson.Encoding qualified as Encoding
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Pair)
import Data.Maybe (isJust)
import Data.Text (Text)
import Invar.Measurement.Duration qualified as Duration
import Invar.Measurement.Resident qualified as Resident
import Invar.Measurement.Stream qualified as Stream
import Invar.Replay.Call qualified as Call
import Invar.Resident.Inference qualified as Physical
import Numeric.Natural (Natural)

data Completion = Completion {completionDigest :: String, wallSeconds :: Double, processSeconds :: Double, physicalOwners :: [Value]}
data Schedule = Finite [(Natural, [Stream.Session])] | Resident Resident.Ledger
data Run = Run
    { logDigest :: String
    , schedule :: Schedule
    , concurrent :: Bool
    , criticalPath :: Double
    , equalResults :: Natural
    , completion :: Maybe Completion
    }

samples :: Run -> [Stream.Sample]
samples = concatMap Stream.requests . measurements

measurements :: Run -> [Stream.Measurement]
measurements run = case schedule run of
    Finite groups -> concatMap (concatMap Stream.measurements . snd) groups
    Resident ledger -> Resident.measurements ledger

loadDurations :: Run -> [Duration.Duration]
loadDurations run = case schedule run of
    Finite groups -> concatMap (map Stream.loading . snd) groups
    Resident ledger -> Resident.loadDurations ledger

costs :: Run -> [Duration.Duration]
costs run = case schedule run of
    Finite _ -> loadDurations run ++ map Stream.inference (measurements run)
    Resident ledger -> Resident.costs ledger

overhead :: Run -> [(Text, [Duration.Duration])]
overhead run = case schedule run of
    Finite _ -> []
    Resident ledger -> Resident.overhead ledger

cohorts :: Run -> Int
cohorts run = case schedule run of
    Finite groups -> length groups
    Resident ledger -> length (Physical.groups (Resident.physical ledger))

counts :: Run -> [Natural]
counts run = case schedule run of
    Finite groups -> map (fromIntegral . length . snd) groups
    Resident ledger -> replicate (cohorts run) (Physical.sessions (Resident.physical ledger))

loadCounts :: Run -> [Natural]
loadCounts run = case schedule run of
    Finite _ -> counts run
    Resident ledger -> Resident.loadCounts ledger

layout :: Run -> [[[Int]]]
layout run = case schedule run of
    Finite groups -> map (map (map (length . Stream.requests) . Stream.measurements) . snd) groups
    Resident ledger -> Physical.layout (Resident.physical ledger)

residentOwners :: Run -> Maybe Natural
residentOwners run = case schedule run of
    Finite _ -> Nothing
    Resident ledger -> Just (Physical.sessions (Resident.physical ledger))

clock :: Run -> Either String Duration.Clock
clock run = case map Duration.clock (costs run) of
    first : rest | all (== first) rest -> pure first
    _ -> Left "Missing or mixed measurement clocks in a run"

paired :: Run -> Run -> Either String Run
paired expected actual = do
    when (isJust (residentOwners expected) || isJust (residentOwners actual)) $
        unless ((residentOwners expected, counts expected, loadCounts expected, layout expected, concurrent expected) == (residentOwners actual, counts actual, loadCounts actual, layout actual, concurrent actual)) (Left "Resident measurements differ from the reference physical lifetimes or numerical grouping")
    let first = samples expected
        second = samples actual
    unless (length first == length second) (Left "Mismatched repeated evaluation inventory")
    let observed = zip first second
    unless (all sameInput observed) (Left "Repeated evaluation consumed different inputs or order")
    unless (all (\(left, right) -> Stream.profile left == Stream.profile right) observed) (Left "Measured numerical profile or model revision differs from the reference")
    firstClock <- clock expected
    secondClock <- clock actual
    unless (firstClock == secondClock) (Left "Measured clock differs from the reference")
    pure actual {equalResults = fromIntegral (length (filter sameResult observed))}
  where
    sameInput (left, right) = let first = Stream.call left; second = Stream.call right in Call.cohort first == Call.cohort second && Call.consumed first == Call.consumed second
    sameResult (left, right) = Call.result (Stream.call left) == Call.result (Stream.call right)

describe :: Run -> Value
describe run =
    object
        ( [ "log_sha256" .= logDigest run
          , "cohorts" .= cohorts run
          , "sessions_per_cohort" .= counts run
          , "concurrent" .= concurrent run
          , "critical_path_seconds" .= criticalPath run
          , "equal_results" .= equalResults run
          ]
            ++ measurementFields run
            ++ lifetimeFields run
            ++ maybe [] completionFields (completion run)
        )

completionFields :: Completion -> [Pair]
completionFields reported = ["completion_sha256" .= completionDigest reported, "campaign_wall_seconds" .= wallSeconds reported, "process_seconds" .= processSeconds reported] ++ ["physical_owners" .= physicalOwners reported | not (null (physicalOwners reported))]

measurementFields :: Run -> [Pair]
measurementFields run = ["measurements" .= map Stream.describe (measurements run), "loads" .= loaded]
  where
    loaded = case schedule run of
        Finite groups -> [withCohort index (Stream.loadValue session) | (index, sessions) <- groups, session <- sessions]
        Resident ledger -> Resident.loadValues ledger
    withCohort index (Object fields) = Object (Fields.insert "cohort" (toJSON index) fields)
    withCohort _ value = value

measurementEncoding :: Run -> Encoding.Series
measurementEncoding run = "measurements" .= measurements run <> Encoding.pair "loads" loaded
  where
    loaded = case schedule run of
        Finite groups -> Encoding.list id [Stream.loadEncoding index session | (index, sessions) <- groups, session <- sessions]
        Resident ledger -> Resident.loadEncoding ledger

lifetimeFields :: Run -> [Pair]
lifetimeFields run = case schedule run of
    Finite _ -> []
    Resident ledger -> Resident.fields ledger
