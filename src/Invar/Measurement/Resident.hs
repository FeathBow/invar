{-# LANGUAGE OverloadedStrings #-}

module Invar.Measurement.Resident (Ledger, admit, physical, samples, measurements, loadDurations, loadCounts, costs, overhead, critical, fields, loadValues, loadEncoding) where

import Control.Monad (unless)
import Data.Aeson (ToJSON (..), Value, object, (.=))
import Data.Aeson.Encoding qualified as Encoding
import Data.Aeson.Types (Pair)
import Data.Maybe (mapMaybe)
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8)
import Invar.Infer.Framing qualified as Frame
import Invar.Measurement.Duration qualified as Duration
import Invar.Measurement.Stream qualified as Stream
import Invar.Resident qualified as Boundary
import Invar.Resident.Inference qualified as Resident
import Invar.Resident.Observation qualified as Observation
import Numeric.Natural (Natural)

data Ledger = Ledger
    { physical :: Resident.Ledger
    , executions :: [(Natural, [(Observation.Group, Stream.Measurement, Double)])]
    , loads :: [Load]
    , critical :: Double
    }

data Load = Load Natural Boundary.Owner Duration.Duration Double [Int]

admit :: Resident.Ledger -> Either String Ledger
admit observed = do
    groups <- traverse group (Resident.groups observed)
    loaded <- traverse (load groups) [(index, observation, duration) | (index, accepted) <- groups, (observation, _, _) <- accepted, duration <- maybe [] pure (Observation.loading observation)]
    let provisional = Ledger observed groups loaded 0
        durations = costs provisional
    case durations of
        first : rest -> unless (all (Duration.sameClock first) rest) (Left "Resident measurement mixes physical process clocks")
        [] -> Left "Missing resident measurements"
    paths <- traverse (schedule . snd) groups
    total <- Duration.total (paths ++ [Duration.seconds duration | (_, _, duration) <- Resident.closes observed])
    pure provisional {critical = total}
  where
    group (index, accepted) = (index,) <$> traverse (execution index) accepted
    execution index observation = do
        measured <- Stream.resident index observation
        elapsed <- Duration.total (map Duration.seconds (Observation.costs observation))
        pure (observation, measured, elapsed)
    load groups (index, observation, duration) = do
        let owner = Observation.physicalOwner observation
            selected = [measured | (_, accepted) <- groups, (actual, measured, _) <- accepted, Observation.physicalOwner actual == owner]
        inference <- Duration.total (map (Duration.seconds . Stream.inference) selected)
        pure (Load index owner duration inference (map (length . Stream.requests) selected))
    schedule accepted
        | Resident.sessions observed > 1 = Duration.checkedSeconds (foldr (\(_, _, elapsed) -> max elapsed) 0 accepted)
        | otherwise = Duration.total [elapsed | (_, _, elapsed) <- accepted]

measurements :: Ledger -> [Stream.Measurement]
measurements ledger = [measured | (_, accepted) <- executions ledger, (_, measured, _) <- accepted]

samples :: Ledger -> [Stream.Sample]
samples = concatMap Stream.requests . measurements

loadDurations :: Ledger -> [Duration.Duration]
loadDurations ledger = [duration | Load _ _ duration _ _ <- loads ledger]

loadCounts :: Ledger -> [Natural]
loadCounts ledger = [fromIntegral (length (mapMaybe (Observation.loading . first) accepted)) | (_, accepted) <- executions ledger]
  where
    first (value, _, _) = value

costs :: Ledger -> [Duration.Duration]
costs ledger = [duration | (_, accepted) <- executions ledger, (observation, _, _) <- accepted, duration <- Observation.costs observation] ++ map snd (closing ledger)

overhead :: Ledger -> [(Text, [Duration.Duration])]
overhead ledger = [("activation", mapMaybe Observation.activation observed), ("release", map Observation.release observed), ("close", map snd (closing ledger))]
  where
    observed = [observation | (_, accepted) <- executions ledger, (observation, _, _) <- accepted]

closing :: Ledger -> [(Frame.Frame, Duration.Duration)]
closing ledger = [(record, duration) | (_, record, duration) <- Resident.closes (physical ledger)]

fields :: Ledger -> [Pair]
fields ledger =
    [ "worker_mode" .= ("resident" :: Text)
    , "model_loads_per_cohort" .= loadCounts ledger
    , "active_sessions_per_cohort" .= map (length . snd) (executions ledger)
    , "resident_groups" .= [object ["cohort" .= index, "observation" .= Observation.describe observation] | (index, accepted) <- executions ledger, (observation, _, _) <- accepted]
    , "closed" .= [object ["source_json" .= decodeUtf8 (Frame.raw record), "duration" .= duration] | (record, duration) <- closing ledger]
    ]

loadFields :: Load -> [Pair]
loadFields (Load index (Boundary.Owner _ owner) _ inference counts) = ["cohort" .= index, "owner" .= owner, "inference_seconds" .= inference, "calls" .= sum counts, "requests_per_execution" .= counts]

instance ToJSON Load where
    toJSON loaded@(Load _ _ duration _ _) = object (loadFields loaded ++ ["load" .= duration])
    toEncoding loaded@(Load _ _ duration _ _) = Encoding.pairs (foldMap (uncurry (.=)) (loadFields loaded) <> "load" .= duration)

loadValues :: Ledger -> [Value]
loadValues = map toJSON . loads

loadEncoding :: Ledger -> Encoding.Encoding
loadEncoding = toEncoding . loads
