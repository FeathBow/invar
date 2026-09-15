{-# LANGUAGE OverloadedStrings #-}

module Invar.Measurement.Statistics (Summary, summarize, describe, comparison) where

import Control.Monad (unless)
import Data.Aeson (ToJSON (..), Value, object, (.=))
import Data.Aeson.Encoding qualified as Encoding
import Data.Aeson.Key (Key)
import Data.Aeson.Key qualified as Key
import Data.Aeson.Types (Pair)
import Data.List (sort)
import Data.Set qualified as Set
import Data.Text (Text)
import Invar.Measurement.Duration qualified as Duration
import Invar.Measurement.Manifest qualified as Manifest
import Invar.Measurement.Run qualified as Run
import Invar.Measurement.Stream qualified as Stream
import Invar.Replay.Call qualified as Call

data Summary = Summary
    { selected :: Manifest.Run
    , observed :: Run.Run
    , measuredClock :: Duration.Clock
    , residual :: Double
    , description :: [Pair]
    , loadStatistics :: DurationSummary
    , inferenceStatistics :: DurationSummary
    , overheadStatistics :: [(Text, Maybe DurationSummary)]
    }

newtype DurationSummary = DurationSummary [(Key, Double)]

instance ToJSON DurationSummary where
    toJSON (DurationSummary values) = object (map (uncurry (.=)) values)
    toEncoding (DurationSummary values) = Encoding.pairs (foldMap (uncurry (.=)) values)

instance ToJSON Summary where
    toJSON = describe
    toEncoding summary = Encoding.pairs (foldMap (uncurry (.=)) (description summary) <> "load_seconds" .= loadStatistics summary <> "inference_seconds" .= inferenceStatistics summary <> foldMap (\(name, value) -> Key.fromText (name <> "_seconds") .= value) (overheadStatistics summary))

summarize :: Manifest.Run -> Run.Run -> Either String Summary
summarize declared run = do
    clock <- Run.clock run
    let samples = Run.samples run
        tokens = sum (map (Call.responseTokens . Stream.call) samples)
        elapsed = Manifest.elapsed declared
        loadDurations = Run.loadDurations run
        inferenceDurations = map Stream.inference (Run.measurements run)
    (load, loadTotal) <- durations (map Duration.seconds loadDurations)
    (inference, inferenceTotal) <- durations (map Duration.seconds inferenceDurations)
    unless (inferenceTotal > 0) (Left "Expected a positive total inference duration")
    peaks <- allocator clock (Run.costs run)
    additional <- traverse summarizeOverhead (Run.overhead run)
    outside <- finite (elapsed - Run.criticalPath run)
    workerTotal <- case Run.residentOwners run of
        Nothing -> finite (loadTotal + inferenceTotal)
        Just _ -> Duration.total (map Duration.seconds (Run.costs run))
    elapsedRate <- finite (fromIntegral tokens / elapsed)
    inferenceRate <- finite (fromIntegral tokens / inferenceTotal)
    let value =
            [ "calls" .= length samples
            , "response_tokens" .= tokens
            , "equal_results" .= Run.equalResults run
            , "worker_seconds_total" .= workerTotal
            , "worker_critical_path_seconds" .= Run.criticalPath run
            , "seconds_outside_worker_critical_path" .= outside
            , "response_tokens_per_elapsed_second" .= elapsedRate
            , "response_tokens_per_inference_second" .= inferenceRate
            , "log_sha256" .= Run.logDigest run
            ]
                ++ peaks
                ++ runFields declared run
                ++ clockFields clock
    pure (Summary declared run clock outside value load inference additional)
  where
    summarizeOverhead (name, []) = pure (name, Nothing)
    summarizeOverhead (name, values) = do
        (summary, _) <- durations (map Duration.seconds values)
        pure (name, Just summary)

runFields :: Manifest.Run -> Run.Run -> [Pair]
runFields declared run =
    [ "name" .= Manifest.name declared
    , "route" .= Manifest.routeValue (Manifest.route declared)
    , "elapsed_seconds" .= Manifest.elapsed declared
    , "cohorts" .= Run.cohorts run
    , "sessions_per_cohort" .= Run.counts run
    , "concurrent" .= Run.concurrent run
    ]
        ++ maybe [] Run.completionFields (Run.completion run)
        ++ maybe [] (const ["worker_mode" .= ("resident" :: Text), "model_loads_per_cohort" .= Run.loadCounts run]) (Run.residentOwners run)

durations :: [Double] -> Either String (DurationSummary, Double)
durations values = do
    total <- Duration.total values
    case sort values of
        [] -> Left "Missing measurement durations"
        ordered@(first : rest) -> do
            middle <- median ordered
            let largest = foldr max first rest
            pure (DurationSummary [("total", total), ("minimum", first), ("median", middle), ("maximum", largest)], total)
  where
    median ordered =
        let count = length ordered
            (left, right) = splitAt (count `div` 2) ordered
         in case (odd count, reverse left, right) of
                (True, _, middle : _) -> pure middle
                (False, before : _, after : _) -> finite ((before + after) / 2)
                _ -> Left "Missing median measurement duration"

allocator :: Duration.Clock -> [Duration.Duration] -> Either String [Pair]
allocator Duration.CpuClock _ = pure []
allocator Duration.MetalClock durationsObserved = do
    values <- maybe (Left "Missing native MLX memory observations") pure (traverse Duration.nativeMemory durationsObserved)
    let active = maximum (0 : map fst values)
        cached = maximum (0 : map snd values)
    pure ["allocator" .= ("mlx" :: Text), "peak_active" .= active, "maximum_cache_at_stage_end" .= cached]
allocator Duration.WorkerClock durationsObserved = do
    values <- maybe (Left "Missing allocator measurements for the reported worker clock") pure (traverse Duration.peaks durationsObserved)
    let (allocated, reserved) = foldr (\(first, second) (largestFirst, largestSecond) -> (max first largestFirst, max second largestSecond)) (0, 0) values
    pure ["peak_allocated" .= allocated, "peak_reserved" .= reserved]

clockFields :: Duration.Clock -> [Pair]
clockFields Duration.WorkerClock = []
clockFields Duration.MetalClock = ["measurement_clock" .= ("mlx_synchronized_seconds" :: Text)]
clockFields Duration.CpuClock = ["measurement_clock" .= ("cpu_seconds" :: Text)]

describe :: Summary -> Value
describe summary = object (description summary ++ ["load_seconds" .= loadStatistics summary, "inference_seconds" .= inferenceStatistics summary] ++ [Key.fromText (name <> "_seconds") .= value | (name, value) <- overheadStatistics summary])

comparison :: [Summary] -> Either String Value
comparison summaries = do
    let firstRoute = filter ((== Manifest.Invar) . Manifest.route . selected) summaries
        secondRoute = filter ((== Manifest.Direct) . Manifest.route . selected) summaries
        requiredRepeats = 2
    unless (length firstRoute >= requiredRepeats && length secondRoute >= requiredRepeats) (Left "Repeated runs required for both routes")
    unless (Set.size (Set.fromList (map (Run.loadCounts . observed) summaries)) == 1) (Left "Matched measurements require the same number of model loads per cohort")
    unless (Set.size (Set.fromList (map (Run.counts . observed) summaries)) == 1) (Left "Matched measurements require the same physical owner count per cohort")
    unless (Set.size (Set.fromList (map (Run.layout . observed) summaries)) == 1) (Left "Matched measurements require the same request groups per numerical execution")
    unless (Set.size (Set.fromList (map signature summaries)) == 1) (Left "Matched measurements require the same session schedule and clock on every run")
    case summaries of
        [] -> Left "Missing repeated measurements"
        first : _ -> do
            firstElapsed <- mean (map (Manifest.elapsed . selected) firstRoute)
            secondElapsed <- mean (map (Manifest.elapsed . selected) secondRoute)
            firstResidual <- mean (map residual firstRoute)
            secondResidual <- mean (map residual secondRoute)
            elapsedDifference <- finite (firstElapsed - secondElapsed)
            residualDifference <- finite (firstResidual - secondResidual)
            pure
                ( object
                    ( [ "repeats" .= object ["invar" .= length firstRoute, "direct" .= length secondRoute]
                      , "cohorts" .= Run.cohorts (observed first)
                      , "sessions_per_cohort" .= Run.counts (observed first)
                      , "concurrent" .= Run.concurrent (observed first)
                      , "mean_elapsed_seconds" .= object ["invar" .= firstElapsed, "direct" .= secondElapsed]
                      , "invar_minus_direct_elapsed_seconds" .= elapsedDifference
                      , "invar_minus_direct_residual_seconds" .= residualDifference
                      , "all_results_equal_to_reference" .= all equal summaries
                      ]
                        ++ clockFields (measuredClock first)
                    )
                )
  where
    signature summary = (Run.concurrent (observed summary), Run.residentOwners (observed summary), measuredClock summary)
    equal summary = Run.equalResults (observed summary) == fromIntegral (length (Run.samples (observed summary)))

mean :: [Double] -> Either String Double
mean [] = Left "Missing repeated measurement values"
mean values = finite (fromRational (sum (map toRational values) / fromIntegral (length values)))

finite :: Double -> Either String Double
finite value
    | isNaN value || isInfinite value = Left "Non-finite measurement statistic"
    | otherwise = pure value
