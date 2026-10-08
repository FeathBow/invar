{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Schedule (Schedule, project, differences, describe) where

import Data.Aeson (Value, object, toJSON, (.=))
import Data.Text (Text)
import Invar.Learn.Request qualified as Request
import Invar.Learn.Stream qualified as S
import Numeric.Natural (Natural)

data Update = Update {update :: Natural, staleness :: Natural, version :: Natural, order :: [Text], steps :: [[Text]]}
    deriving (Eq, Show)

newtype Schedule = Schedule [Update]
    deriving (Eq, Show)

project :: [Request.Request] -> Schedule
project = Schedule . map scheduled
  where
    scheduled request = Update (Request.scheduled request) (Request.staleness request) (Request.version request) (Request.order request) (S.batches (Request.exchange request))

differences :: Schedule -> Schedule -> [Value]
differences (Schedule left) (Schedule right) =
    [object ["generation" .= index, "fields" .= changed] | (index, first, second) <- zip3 [1 :: Natural ..] left right, let changed = fields first second, not (null changed)]
        ++ [object ["generation" .= index, "missing" .= ("left" :: Text)] | (index, _) <- drop (length left) (zip [1 :: Natural ..] right)]
        ++ [object ["generation" .= index, "missing" .= ("right" :: Text)] | (index, _) <- drop (length right) (zip [1 :: Natural ..] left)]
  where
    fields first second = [name | (name, same) <- [("update", update first == update second), ("staleness", staleness first == staleness second), ("version", version first == version second), ("order", order first == order second), ("steps", steps first == steps second)], not same] :: [Text]

describe :: Schedule -> Value
describe (Schedule updates) = toJSON [object ["update" .= update scheduled, "staleness" .= staleness scheduled, "version" .= version scheduled, "order" .= order scheduled, "steps" .= steps scheduled] | scheduled <- updates]
