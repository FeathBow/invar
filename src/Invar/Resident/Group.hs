{-# LANGUAGE OverloadedStrings #-}

module Invar.Resident.Group (Group, observe, physicalOwner, ordinal, prefix, bindings, describe) where

import Data.Aeson (Value (..), object, (.=))
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8)
import Invar.Infer.Framing qualified as Frame
import Invar.Measurement.Duration qualified as Duration
import Invar.Resident qualified as Boundary
import Invar.Spec.Invocation qualified as Invocation
import Numeric.Natural (Natural)

data Group = Group
    { physicalOwner :: Boundary.Owner
    , ordinal :: Natural
    , prefix :: [Frame.Frame]
    , body :: [Frame.Frame]
    , bindings :: [Invocation.Binding]
    , acknowledgement :: Frame.Frame
    , loading :: Maybe Duration.Duration
    , activation :: Maybe Duration.Duration
    , release :: Duration.Duration
    , costs :: [Duration.Duration]
    }
    deriving (Eq, Show)

observe :: (Boundary.Owner, Natural, [Invocation.Binding]) -> [Frame.Frame] -> Frame.Frame -> Either String Group
observe (selected, index, bound) records acknowledged = do
    let (leading, execution) = span ((`elem` map (Just . String) ["loading", "profile", "load", "activation"]) . Frame.stageName) records
    initialLoad <- measurement "load" leading
    selectedActivation <- measurement "activation" leading
    released <- Boundary.measured "released" (Frame.raw acknowledged)
    measured <- traverse timing (filter ((`elem` map (Just . String) ["load", "activation", "inference", "reward_update", "artifacts", "checkpoint"]) . Frame.stageName) records)
    pure (Group selected index leading execution bound acknowledged initialLoad selectedActivation released (measured ++ [released]))

measurement :: Text -> [Frame.Frame] -> Either String (Maybe Duration.Duration)
measurement name records = case filter ((== Just (String name)) . Frame.stageName) records of
    [] -> pure Nothing
    [record] -> Just <$> timing record
    _ -> Left "Repeated resident operation measurement"

timing :: Frame.Frame -> Either String Duration.Duration
timing record = Duration.admit (Frame.raw record) (Frame.fields record)

describe :: Group -> Value
describe group = object ["owner" .= Boundary.ownerValue (physicalOwner group), "group" .= ordinal group, "source_jsonl" .= decodeUtf8 (Frame.encode (prefix group ++ body group)), "released_json" .= decodeUtf8 (Frame.raw (acknowledgement group)), "load" .= loading group, "activation" .= activation group, "release" .= release group, "costs" .= costs group]
