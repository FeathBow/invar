{-# LANGUAGE OverloadedStrings #-}

module Invar.Measurement.Inference (Report, admit, describe, replayReference, observedRun) where

import Control.Monad (unless)
import Data.Aeson (Object, ToJSON (..), Value (..), object, withObject, (.:), (.:?), (.=))
import Data.Aeson.Encoding qualified as Encoding
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Pair, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Map.Strict qualified as Map
import Invar.Evaluation qualified as Evaluation
import Invar.Json qualified as Json
import Invar.Measurement.Duration qualified as Duration
import Invar.Measurement.Resident qualified as Resident
import Invar.Measurement.Run qualified as Run
import Invar.Measurement.Stream qualified as Stream
import Invar.Replay.Call qualified as Call
import Invar.Replay.Inference qualified as Replay
import Invar.Resident.Inference qualified as Physical
import Invar.Spec.Invocation qualified as V
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)

data Report = Report Replay.Reference Run.Run

replayReference :: Report -> Replay.Reference
replayReference (Report reference _) = reference

observedRun :: Report -> Run.Run
observedRun (Report _ observed) = observed

admit :: Workload.Document -> Evaluation.Run -> ByteString -> Either String Report
admit tasks declared encoded = do
    reference <- Replay.admit tasks (declared, Replay.Process) encoded
    (selected, concurrent, critical) <- case Evaluation.residence (Replay.evaluation reference) of
        Nothing -> do
            (concurrent, measured, critical) <- finite encoded
            pure (Run.Finite (zip [0 ..] measured), concurrent, critical)
        Just physical -> do
            measured <- Resident.admit physical
            pure (Run.Resident measured, Physical.sessions physical > 1, Resident.critical measured)
    let observed = Run.Run {Run.logDigest = Evaluation.logDigest (Replay.evaluation reference), Run.schedule = selected, Run.concurrent = concurrent, Run.criticalPath = critical, Run.equalResults = fromIntegral (length (Replay.calls reference)), Run.completion = Nothing}
        declaredSamples = Map.fromList [(V.boundCall (Evaluation.sampleBinding sample), sample) | sample <- Evaluation.samples (Replay.evaluation reference)]
    unless (length (Run.samples observed) == length (Replay.calls reference)) (Left "Incomplete evaluation measurement inventory")
    mapM_ (checkSummary declaredSamples . Stream.call) (Run.samples observed)
    pure (Report reference observed)

finite :: ByteString -> Either String (Bool, [[Stream.Session]], Double)
finite encoded = do
    frames <- traverse (\raw -> (,) raw <$> (Json.decode raw >>= parseEither (withObject "measured evaluation" pure))) (Bytes.lines encoded)
    declared <- case reverse frames of
        (_, completed) : _ -> parseEither (.:? "sessions") completed
        [] -> Left "Missing evaluation completion"
    chunks <- cohorts frames
    measured <- traverse (uncurry Stream.admit) (zip [0 ..] chunks)
    let counts = map (fromIntegral . length) measured :: [Natural]
    unless (maybe True (\count -> all (== count) counts) declared) (Left "Declared session count differs from measured model sessions")
    let concurrent = maybe False (> 1) declared
    paths <- traverse (schedule concurrent) measured
    critical <- Duration.checkedSeconds (foldl' (+) 0 paths)
    pure (concurrent, measured, critical)

schedule :: Bool -> [Stream.Session] -> Either String Double
schedule concurrent sessions
    | concurrent = Duration.checkedSeconds (foldr (max . Stream.elapsed) 0 sessions)
    | otherwise = Duration.total (map Stream.elapsed sessions)

cohorts :: [(ByteString, Object)] -> Either String [ByteString]
cohorts = collect [] []
  where
    collect completed [] [] = pure (reverse completed)
    collect _ _ [] = Left "Worker records after the final evaluation cohort"
    collect completed pending ((raw, fields) : rest)
        | Fields.lookup "phase" fields == Just (String "evaluation") = collect (Bytes.unlines (reverse pending) : completed) [] rest
        | Fields.member "stage" fields = collect completed (raw : pending) rest
        | otherwise = collect completed pending rest

checkSummary :: Map.Map V.CallId Evaluation.Sample -> Call.Call -> Either String ()
checkSummary declared observed = do
    sample <- maybe (Left "Measured result has no evaluation sample") Right (Map.lookup (V.boundCall (Call.bound observed)) declared)
    truncated <- parseEither (.: "truncated") (Call.result observed)
    unless (Call.cohort observed == Evaluation.sampleCohort sample && Call.bound observed == Evaluation.sampleBinding sample && Call.responseTokens observed == Evaluation.sampleTokens sample && truncated == Evaluation.sampleTruncated sample) (Left "Worker result differs from evaluation summary")

describe :: Report -> Value
describe report = object (metadata report ++ Run.measurementFields (observedRun report) ++ Run.lifetimeFields (observedRun report))

instance ToJSON Report where
    toJSON = describe
    toEncoding report = Encoding.pairs (foldMap (uncurry (.=)) (metadata report ++ Run.lifetimeFields (observedRun report)) <> Run.measurementEncoding (observedRun report))

metadata :: Report -> [Pair]
metadata (Report reference observed) =
    [ "log_sha256" .= Evaluation.logDigest evaluated
    , "tasks_sha256" .= Evaluation.inputDigest evaluated
    , "cohorts" .= Run.cohorts observed
    , "sessions_per_cohort" .= Run.counts observed
    , "concurrent" .= Run.concurrent observed
    , "critical_path_seconds" .= Run.criticalPath observed
    ]
  where
    evaluated = Replay.evaluation reference
