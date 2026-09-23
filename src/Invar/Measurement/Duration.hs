{-# LANGUAGE OverloadedStrings #-}

module Invar.Measurement.Duration (Duration, Clock (..), admit, value, encoding, seconds, clock, peaks, total) where

import Control.Monad (unless)
import Data.Aeson (Object, ToJSON (..), Value (..), object, (.:), (.=))
import Data.Aeson.Encoding qualified as Encoding
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Invar.Json qualified as Json
import Numeric.Natural (Natural)

data Duration = Worker Double Natural Natural | Metal Double Natural Natural | Cpu Double deriving (Eq, Show)
data Clock = WorkerClock | MetalClock | CpuClock deriving (Eq, Ord, Show)

admit :: ByteString -> Object -> Either String Duration
admit raw fields = do
    elapsed <- Json.floatingAt [if Fields.member "cpu_seconds" fields then "cpu_seconds" else "seconds"] raw
    parseEither (parse elapsed) fields

parse :: Double -> Object -> Parser Duration
parse elapsed fields
    | Fields.member "cpu_seconds" fields = do
        Json.fields ["stage", "cpu_seconds"] fields
        unless (elapsed >= 0) (fail "Negative CPU duration")
        pure (Cpu elapsed)
    | Fields.member "allocator" fields = do
        Json.fields ["stage", "seconds", "allocator", "peak_active", "cache_end"] fields
        unless (Fields.lookup "allocator" fields == Just (String "mlx") && elapsed >= 0) (fail "Invalid native MLX measurement")
        Metal elapsed <$> fields .: "peak_active" <*> fields .: "cache_end"
    | otherwise = do
        Json.fields ["stage", "seconds", "peak_allocated", "peak_reserved"] fields
        unless (elapsed >= 0) (fail "Negative worker duration")
        Worker elapsed <$> fields .: "peak_allocated" <*> fields .: "peak_reserved"

value :: Duration -> Value
value (Worker elapsed allocated reserved) = object ["seconds" .= elapsed, "peak_allocated" .= allocated, "peak_reserved" .= reserved]
value (Metal elapsed active cached) = object ["seconds" .= elapsed, "allocator" .= String "mlx", "peak_active" .= active, "cache_end" .= cached]
value (Cpu elapsed) = object ["cpu_seconds" .= elapsed]

instance ToJSON Duration where
    toJSON = value
    toEncoding = Encoding.pairs . encoding

encoding :: Duration -> Encoding.Series
encoding (Worker elapsed allocated reserved) = "seconds" .= elapsed <> "peak_allocated" .= allocated <> "peak_reserved" .= reserved
encoding (Metal elapsed active cached) = "seconds" .= elapsed <> "allocator" .= String "mlx" <> "peak_active" .= active <> "cache_end" .= cached
encoding (Cpu elapsed) = "cpu_seconds" .= elapsed

seconds :: Duration -> Double
seconds (Worker elapsed _ _) = elapsed
seconds (Metal elapsed _ _) = elapsed
seconds (Cpu elapsed) = elapsed

clock :: Duration -> Clock
clock (Worker {}) = WorkerClock
clock (Metal {}) = MetalClock
clock (Cpu _) = CpuClock

peaks :: Duration -> Maybe (Natural, Natural)
peaks (Worker _ allocated reserved) = Just (allocated, reserved)
peaks _ = Nothing

total :: [Double] -> Either String Double
total values = checkedSeconds (if correction /= 0 && not (isInfinite correction || isNaN correction) then high + correction else high)
  where
    (high, correction) = foldl' add (0, 0) values
    add (running, low) next =
        let combined = running + next
            lost = if abs running >= abs next then (running - combined) + next else (next - combined) + running
         in (combined, low + lost)

checkedSeconds :: Double -> Either String Double
checkedSeconds elapsed
    | isNaN elapsed || isInfinite elapsed || elapsed < 0 = Left "Non-finite or negative aggregate measurement duration"
    | otherwise = Right elapsed
