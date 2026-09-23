{-# LANGUAGE OverloadedStrings #-}

module Invar.Measurement.Stream (Session, Sample, Measurement, admit, resident, samples, measurements, loading, elapsed, inference, requests, call, profile, describe, loadValue, loadEncoding) where

import Control.Monad (unless, when)
import Data.Aeson (Object, ToJSON (..), Value (..), object, withObject, (.=))
import Data.Aeson.Encoding qualified as Encoding
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Pair, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Measurement.Duration qualified as Duration
import Invar.Measurement.Profile qualified as Profile
import Invar.Replay.Call qualified as Call
import Invar.Replay.Load qualified as Load
import Invar.Resident.Observation qualified as Resident
import Numeric.Natural (Natural)

data Session = Session {loading :: Duration.Duration, measurements :: [Measurement], inferenceSeconds :: Double, elapsed :: Double}
data Measurement = Serial Duration.Duration Sample | Batched Duration.Duration [Sample]
data Sample = Sample {call :: Call.Call, profile :: String}
data Frame = Frame ByteString Object

admit :: Natural -> ByteString -> Either String [Session]
admit cohort encoded = do
    unless ("\n" `Bytes.isSuffixOf` encoded) (Left "Incomplete measurement record stream")
    frames <- traverse (\raw -> Frame raw <$> (Json.decode raw >>= parseEither (withObject "measurement record" pure))) (Bytes.lines encoded)
    when (null frames) (Left "Missing measured model sessions")
    sessions frames
  where
    sessions [] = pure []
    sessions frames = do
        let afterLoading = case frames of
                first : rest | stage first == Just (String "loading") -> rest
                _ -> frames
        (reported, duration, execution) <- case afterLoading of
            first@(Frame rawProfile fields) : second@(Frame rawDuration elapsed) : rest | stage first == Just (String "profile") && stage second == Just (String "load") -> do
                when (Fields.member "phase" fields) (Left "Unexpected phase in numerical profile")
                checked <- Duration.admit rawDuration elapsed
                canonical <- Profile.admit rawProfile
                pure (canonical, checked, rest)
            _ -> Left "Model load without its own preceding profile"
        let (blocks, remaining) = break (\frame -> stage frame `elem` map (Just . String) ["loading", "profile"]) execution
        observed <- blocksOf reported Nothing blocks
        when (null observed) (Left "Model session without results")
        unless (all (Duration.sameClock duration . inference) observed) (Left "Mixed measurement clocks inside one session")
        inferenceTotal <- Duration.total (map (Duration.seconds . inference) observed)
        sessionTotal <- Duration.checkedSeconds (Duration.seconds duration + inferenceTotal)
        rest <- sessions remaining
        pure (Session duration observed inferenceTotal sessionTotal : rest)
    blocksOf _ _ [] = pure []
    blocksOf reported Nothing frames@(first : _) | Framing.grouped (convert first) = do
        (group, rest) <- Framing.takeGroup (map convert frames)
        unless (null rest) (Left "Output follows the finite batch in a measured model session")
        observed <- traverse (\member -> admitSample (cohort, reported) (Framing.fields (Framing.loaded member)) (Framing.raw (Framing.consumed member), Framing.raw (Framing.result member))) (Framing.members group)
        let Framing.Frame rawDuration fields = Framing.duration group
        measured <- Duration.admit rawDuration fields
        pure [Batched measured observed]
    blocksOf reported previous frames = do
        execution <- case frames of
            first@(Frame _ fields) : rest | stage first == Just (String "unloaded_adapter") -> do
                preceding <- maybe (Left "Adapter replacement outside a completed result block") Right previous
                parseEither (Load.unloaded preceding) fields
                pure rest
            _ -> pure frames
        case execution of
            loaded@(Frame _ materialized) : consumed@(Frame rawConsumed _) : timed@(Frame rawDuration elapsed) : result@(Frame rawResult _) : rest -> do
                unless (map stage [loaded, consumed, timed, result] == map (Just . String) ["loaded_adapter", "consumed", "inference", "result"]) (Left "Missing, repeated or reordered worker stages in a result block")
                observed <- admitSample (cohort, reported) materialized (rawConsumed, rawResult)
                measured <- Duration.admit rawDuration elapsed
                following <- blocksOf reported (Just (call observed)) rest
                pure (Serial measured observed : following)
            _ -> Left "Incomplete worker measurement block"
    convert (Frame raw fields) = Framing.Frame raw fields

admitSample :: (Natural, Profile.Profile) -> Object -> (ByteString, ByteString) -> Either String Sample
admitSample (cohort, reported) materialized sources = do
    observed <- Call.admit cohort sources
    identity <- parseEither (Load.admit observed) materialized
    fingerprint <- Profile.fingerprint reported identity
    pure (Sample observed fingerprint)

resident :: Natural -> Resident.Group -> Either String Measurement
resident cohort observed = do
    reported <- case Resident.modelProfiles observed of
        [record] -> Profile.admit (Framing.raw record)
        _ -> Left "Resident measurement requires the original physical model profile"
    (group, remaining) <- Framing.takeGroup (Resident.body observed)
    unless (null remaining) (Left "Trailing resident numerical measurements")
    accepted <- traverse (\member -> admitSample (cohort, reported) (Framing.fields (Framing.loaded member)) (Framing.raw (Framing.consumed member), Framing.raw (Framing.result member))) (Framing.members group)
    let Framing.Frame rawDuration fields = Framing.duration group
    elapsed <- Duration.admit rawDuration fields
    pure (Batched elapsed accepted)

stage :: Frame -> Maybe Value
stage (Frame _ fields) = Fields.lookup "stage" fields

samples :: Session -> [Sample]
samples = concatMap requests . measurements

requests :: Measurement -> [Sample]
requests (Serial _ sample) = [sample]
requests (Batched _ observed) = observed

inference :: Measurement -> Duration.Duration
inference (Serial duration _) = duration
inference (Batched duration _) = duration

sampleFields :: Sample -> [Pair]
sampleFields sample = ["response_tokens" .= Call.responseTokens (call sample), "profile_sha256" .= profile sample]

describe :: Measurement -> Value
describe (Serial measured sample) = object ("inference" .= Duration.value measured : sampleFields sample)
describe (Batched measured observed) = object ["inference" .= Duration.value measured, "calls" .= map sampleValue observed]

sampleValue :: Sample -> Value
sampleValue sample = object ("binding" .= Wire.bindingValue (Call.bound (call sample)) : sampleFields sample)

instance ToJSON Measurement where
    toJSON = describe
    toEncoding (Serial measured sample) = Encoding.pairs ("inference" .= measured <> foldMap (uncurry (.=)) (sampleFields sample))
    toEncoding (Batched measured observed) = Encoding.pairs ("inference" .= measured <> "calls" .= map sampleValue observed)

loadEncoding :: Natural -> Session -> Encoding.Encoding
loadEncoding index session = Encoding.pairs ("cohort" .= index <> "inference_seconds" .= inferenceSeconds session <> "calls" .= length (samples session) <> "requests_per_execution" .= map (length . requests) (measurements session) <> duration)
  where
    duration = Duration.encoding (loading session)

loadValue :: Session -> Value
loadValue session = case Duration.value (loading session) of
    Object fields -> Object (Fields.union fields (Fields.fromList ["inference_seconds" .= inferenceSeconds session, "calls" .= length (samples session), "requests_per_execution" .= map (length . requests) (measurements session)]))
    value -> value
