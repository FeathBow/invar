{-# LANGUAGE OverloadedStrings #-}

module Invar.Score.Resources (observe) where

import Control.Monad (filterM, unless)
import Data.Aeson (Object, Value, withObject, (.:))
import Data.Aeson.Types (Parser)
import Data.Set qualified as Set
import Invar.Json qualified as Json
import Invar.Score.Probe (Relation (..))

observe :: Relation -> [Object] -> Parser ()
observe relation records = do
    scored <- filterM (\fields -> (== ("cross_score" :: String)) <$> fields .: "stage") records
    case scored of
        [fields] -> case relation of
            LogOfMass -> mlx fields
            SeparateLogSoftmax -> native fields
        _ -> fail "Expected exactly one native probe cost measurement"

duration :: Object -> Parser ()
duration fields = do
    seconds <- fields .: "seconds" >>= Json.finite
    unless (seconds >= 0) (fail "Probe duration must be finite and nonnegative")

mlx :: Object -> Parser ()
mlx fields = do
    Json.fields ["stage", "seconds", "allocator", "peak_active", "cache_end"] fields
    duration fields
    allocator <- fields .: "allocator"
    peak <- fields .: "peak_active"
    cache <- fields .: "cache_end"
    unless (allocator == ("mlx" :: String) && peak > (0 :: Integer) && cache >= (0 :: Integer)) (fail "Invalid native MLX probe cost measurement")

native :: Object -> Parser ()
native fields = do
    Json.fields ["stage", "seconds", "seconds_scope", "allocator", "workers"] fields
    duration fields
    scope <- fields .: "seconds_scope"
    allocator <- fields .: "allocator"
    unless (scope == ("client wait_for_completion/v1" :: String) && allocator == ("torch.cuda" :: String)) (fail "Invalid native probe client measurement scope")
    encoded <- fields .: "workers"
    workers <- traverse worker encoded
    unless (not (null workers) && Set.size (Set.fromList workers) == length workers) (fail "Missing or repeated native worker resource identities")

worker :: Value -> Parser (String, Integer, String)
worker = withObject "native worker resources" $ \fields -> do
    Json.fields ["host", "pid", "device", "seconds", "peak_allocated", "peak_reserved", "scope"] fields
    duration fields
    host <- fields .: "host"
    pid <- fields .: "pid"
    device <- fields .: "device"
    unless (not (null host) && pid > 0 && not (null device)) (fail "Missing native worker host/process/device identity")
    scope <- fields .: "scope"
    unless (scope == ("native worker permit-to-completion; loading and serialization excluded/v1" :: String)) (fail "Unexpected native worker resource interval")
    allocated <- fields .: "peak_allocated"
    reserved <- fields .: "peak_reserved"
    unless (0 < (allocated :: Integer) && allocated <= reserved) (fail "Invalid native worker allocation peaks")
    pure (host, pid, device)
