{-# LANGUAGE OverloadedStrings #-}

module Invar.Measurement.Direct.Timing (Interval (..), Status (..), interval, status, validate) where

import Control.Monad (unless)
import Data.Aeson (Object, (.:))
import Data.Aeson.Key (Key)
import Data.Aeson.Types (Parser)
import Data.List (nub)
import Data.Text (Text)
import Invar.Json qualified as Json
import Invar.Measurement.Direct.Record (positive)
import Invar.Measurement.Duration qualified as Duration
import Numeric.Natural (Natural)

data Interval = Interval {identity :: Natural, start :: Double, end :: Double}
data Status = Status {lifetime :: Interval, process :: Double, pid :: Natural, device :: Maybe Text}

interval :: Key -> Object -> Parser Interval
interval key fields = do
    Json.fields [key, "start_offset_seconds", "end_offset_seconds"] fields
    parseInterval key fields

parseInterval :: Key -> Object -> Parser Interval
parseInterval key fields = do
    index <- fields .: key
    started <- fields .: "start_offset_seconds" >>= Json.finite
    finished <- positive "end_offset_seconds" fields
    unless (started >= 0 && finished > started) (fail "Invalid direct physical time interval")
    pure (Interval index started finished)

status :: (Natural, Int) -> Object -> Parser Status
status (owner, calls) fields = do
    Json.fields ["owner", "pid", "device", "exit_code", "process_seconds", "start_offset_seconds", "end_offset_seconds", "calls", "stdout_sha256", "stderr_sha256"] fields
    observed <- parseInterval "owner" fields
    code <- fields .: "exit_code" :: Parser Int
    count <- fields .: "calls" :: Parser Int
    processId <- fields .: "pid"
    unless (identity observed == owner && code == 0 && count == calls && processId > 0) (fail "Failed or incomplete direct resident owner")
    elapsed <- positive "process_seconds" fields
    unless (elapsed == end observed - start observed) (fail "Resident process duration differs from its physical lifetime")
    Status observed elapsed processId <$> fields .: "device"

validate :: (Double, Double) -> ([Interval], [Interval]) -> [Status] -> Either String ()
validate (wall, elapsed) (cohorts, closes) owners = do
    let intervals = cohorts ++ closes
        processes = map lifetime owners
        boundaries = [(end first, start next) | (first, next) <- zip intervals (drop 1 intervals)]
    unless (not (null cohorts) && not (null closes) && all (uncurry (<=)) boundaries && all ((<= wall) . end) intervals) (Left "Resident cohort barriers or ordered shutdown intervals overlap or exceed the campaign")
    case cohorts of
        first : _ -> unless (all ((<= start first) . start) processes) (Left "Resident physical owners did not start before the workload")
        [] -> Left "Missing resident cohort intervals"
    unless (map identity closes == reverse (map identity processes) && map end closes == reverse (map end processes)) (Left "Resident owners did not remain alive through their declared ordered shutdown")
    unless (length (nub (map pid owners)) == length owners) (Left "Repeated direct physical process identity")
    total <- Duration.checkedSeconds (foldl' (\running owner -> running + process owner) 0 owners)
    unless (elapsed == total) (Left "Direct resident process total differs from its physical lifetimes")
