{-# LANGUAGE OverloadedStrings #-}

module Invar.Replay.Resident (Output, admit, groups, closed, closing, owner, rows, equalResults, responseTokens, describe) where

import Control.Monad (foldM, unless)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value (..), object, (.=))
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8)
import Invar.Artifact qualified as Artifact
import Invar.Infer.Framing qualified as Frame
import Invar.Infer.Wire qualified as Wire
import Invar.Measurement.Duration qualified as Duration
import Invar.Replay.Call qualified as Call
import Invar.Replay.Load qualified as Load
import Invar.Resident qualified as Boundary
import Invar.Resident.Observation qualified as Resident
import Numeric.Natural (Natural)

data Output = Output
    { owner :: Natural
    , digest :: String
    , groups :: [(Natural, Resident.Group)]
    , closed :: Frame.Frame
    , closing :: Duration.Duration
    , comparisons :: [(Natural, Natural, Call.Call, Call.Call)]
    }

admit :: (Natural, Int) -> [Call.Call] -> ByteString -> Either String Output
admit (physical, exitCode) expected encoded = do
    unless (exitCode == 0) (Left "Direct resident worker did not exit successfully")
    mapM_ Call.session expected
    let planned = Call.cohorts expected
        indices = map fst planned
    unless (and (zipWith (<) indices (drop 1 indices))) (Left "Resident replay cohorts are repeated or reordered")
    frames <- Frame.decode encoded
    let initial = Resident.empty (Boundary.Owner Boundary.Inference physical)
    (current, accepted, compared, remaining) <- foldM step (initial, [], [], frames) planned
    (lastFrame, elapsed, rest) <- Resident.finish current remaining
    unless (null rest) (Left "Trailing direct resident worker output")
    pure (Output physical (Artifact.hex (SHA256.hash encoded)) (reverse accepted) lastFrame elapsed (concat (reverse compared)))
  where
    step (current, accepted, compared, frames) (cohort, calls) = do
        (next, observed, remaining) <- Resident.inference (cohort, current) frames
        (batch, rest) <- Frame.takeGroup (Resident.body observed)
        unless (null rest && length calls == length (Frame.members batch)) (Left "Resident replay differs from its complete physical group inventory")
        actual <- traverse match (zip calls (Frame.members batch))
        let paired = [(Resident.ordinal observed, index, original, result) | (index, (original, result)) <- zip [0 ..] actual]
        pure (next, (cohort, observed) : accepted, paired : compared, remaining)
    match (planned, member) = do
        _ <- parseEither (Load.admit planned) (Frame.fields (Frame.loaded member))
        let consumed = Frame.consumed member
        unless (Frame.fields consumed == Call.consumed planned) (Left "Direct resident worker consumed different input or group order")
        actual <- Call.admit (Call.cohort planned) (Frame.raw consumed, Frame.raw (Frame.result member))
        pure (planned, actual)

rows :: Output -> [Value]
rows output = [object ["owner" .= owner output, "cohort" .= Call.cohort expected, "group" .= group, "index" .= index, "binding" .= Wire.bindingValue (Call.bound actual), "response_tokens" .= Call.responseTokens actual, "result_equal" .= (Call.result expected == Call.result actual)] | (group, index, expected, actual) <- comparisons output]

equalResults :: Output -> Natural
equalResults output = fromIntegral (length [() | (_, _, expected, actual) <- comparisons output, Call.result expected == Call.result actual])

responseTokens :: Output -> Natural
responseTokens output = sum [Call.responseTokens actual | (_, _, _, actual) <- comparisons output]

describe :: Output -> Value
describe output = object ["mode" .= ("resident" :: Text), "owner" .= owner output, "stdout_sha256" .= digest output, "groups" .= [object ["cohort" .= index, "observation" .= Resident.describe group] | (index, group) <- groups output], "closed_json" .= decodeUtf8 (Frame.raw (closed output)), "close" .= closing output, "calls" .= rows output, "equal_results" .= equalResults output, "response_tokens" .= responseTokens output]
