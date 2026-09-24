{-# LANGUAGE OverloadedStrings #-}

module Invar.Score.Probe (Relation (..), selection, observe) where

import Control.Monad (unless, when, zipWithM)
import Data.Aeson (Value, withObject, (.:))
import Data.Aeson.Types (Parser)
import Data.Map.Strict qualified as Map
import Data.Word (Word32)
import Invar.Json qualified as Json
import Invar.Numerical.Distribution qualified as Distribution
import Invar.Spec.Score qualified as S
import Numeric.Natural (Natural)

data Relation = LogOfMass | SeparateLogSoftmax

selection :: Natural -> [Natural] -> Either String ()
selection horizon steps = do
    when (null steps) (Left "Full-vocabulary probe steps must be nonempty")
    unless (and (zipWith (<) steps (drop 1 steps))) (Left "Probe steps must be strictly increasing")
    unless (all (< horizon) steps) (Left "Probe steps must lie within the actual source response")

observe :: Relation -> ([Natural], [(Natural, Word32)]) -> Value -> Parser S.Distribution
observe relation (steps, path) = withObject "full-vocabulary snapshots" $ \fields -> do
    Json.fields ["steps", "vocabulary", "coordinates", "representation", "snapshots", "raw_payload_bytes"] fields
    actual <- fields .: "steps"
    unless (actual == steps) (fail "Full-vocabulary steps differ from the score plan")
    width <- fields .: "vocabulary"
    unless (width > 0) (fail "The full vocabulary must be nonempty")
    coordinates <- fields .: "coordinates"
    representation <- fields .: "representation"
    unless (coordinates == ("output token ids 0..vocabulary-1" :: String) && representation == ("F32 probability words" :: String)) (fail "Unexpected distribution coordinates or representation")
    encoded <- fields .: "snapshots"
    unless (length encoded == length steps) (fail "Incomplete full-vocabulary snapshot inventory")
    retained <- zipWithM (snapshot (relation, width, Map.fromList (zip [0 ..] path))) steps encoded
    bytes <- fields .: "raw_payload_bytes"
    let fp32Bytes = 4
    unless (bytes == toInteger (length steps) * toInteger width * fp32Bytes) (fail "Incorrect full-vocabulary payload size")
    pure (S.Distribution width retained)

snapshot :: (Relation, Natural, Map.Map Natural (Natural, Word32)) -> Natural -> Value -> Parser S.Snapshot
snapshot (relation, width, path) expected = withObject "full-vocabulary snapshot" $ \fields -> do
    Json.fields ["step", "probability_bits"] fields
    step <- fields .: "step"
    unless (step == expected) (fail "Missing or reordered probe snapshot")
    encodedMasses <- fields .: "probability_bits"
    unless (fromIntegral (length encodedMasses) == width) (fail "Snapshot does not cover the complete vocabulary")
    values <- traverse mass encodedMasses
    unless (any (> 0) values) (fail "A distribution must have positive total mass")
    (token, logWord) <- maybe (fail "Probe step is outside the scored path") pure (Map.lookup expected path)
    selected <- maybe (fail "Scored token is outside the retained vocabulary") pure (lookup token (zip [0 ..] values))
    case relation of
        LogOfMass -> unless ((selected == 0) == (logWord == 0xff800000)) (fail "Selected-token support differs between the vector and path score")
        SeparateLogSoftmax -> pure ()
    pure (S.Snapshot step encodedMasses)

mass :: Word32 -> Parser Integer
mass = either fail pure . Distribution.mass
