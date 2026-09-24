{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Framing (Frame (..), Group, Member (..), stageName, format, decode, encode, grouped, readiness, activationReadiness, completion, takeGroup, groups, members, duration, source, memberBytes) where

import Control.Monad (unless, void, when)
import Data.Aeson (Object, Value (..), withObject, (.:))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text.Encoding (encodeUtf8)
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Measurement.Duration qualified as Duration
import Invar.Spec.Invocation qualified as Invocation

data Frame = Frame {raw :: ByteString, fields :: Object}
    deriving (Eq, Show)
data Member = Member {loaded :: Frame, consumed :: Frame, result :: Frame}
data Group = Group {members :: [Member], duration :: Frame, source :: ByteString}

format :: Text
format = "invar-inference-batch-v1"

decode :: ByteString -> Either String [Frame]
decode encoded = do
    unless ("\n" `Bytes.isSuffixOf` encoded) (Left "Incomplete finite batch observation line")
    traverse (\raw -> Frame raw <$> (Json.decode raw >>= parseEither (withObject "finite batch observation" pure))) (Bytes.lines encoded)

encode :: [Frame] -> ByteString
encode = Bytes.unlines . map raw

stageName :: Frame -> Maybe Value
stageName = Fields.lookup "stage" . fields

grouped :: Frame -> Bool
grouped (Frame _ fields) = Fields.member "format" fields && Fields.lookup "stage" fields `elem` map (Just . String) ["consumed", "result"]

readiness :: [Frame] -> Either String [ByteString]
readiness = readyWith loading

activationReadiness :: [Frame] -> Either String [ByteString]
activationReadiness = readyWith activation
  where
    activation [record] = stage "activation" record >> timing record
    activation _ = Left "Expected one actual resident activation measurement"

readyWith :: ([Frame] -> Either String ()) -> [Frame] -> Either String [ByteString]
readyWith prefix records = case reverse records of
    Frame _ consumed : preceding -> do
        prefix (reverse preceding)
        values <- payload "consumed" consumed
        mapM_ (member ["loaded_adapter", "consumed"]) values
        pure values
    [] -> Left "Missing finite batch consumption"

completion :: [Frame] -> Either String (Frame, [ByteString])
completion records = case records of
    [measured, Frame _ finished] -> do
        stage "inference" measured
        timing measured
        values <- payload "result" finished
        mapM_ (member ["result"]) values
        pure (measured, values)
    _ -> Left "Expected one actual batch inference measurement and complete result"

loading :: [Frame] -> Either String ()
loading records = do
    let stages = map (Fields.lookup "stage" . fields) records
    unless (stages `elem` map (map (Just . String)) [["load"], ["loading", "profile", "load"]]) (Left "Expected one ordered model load for the finite batch")
    mapM_ (\record -> when (Fields.member "phase" (fields record)) (Left "Unexpected phase in batch loading observations")) records
    case reverse records of
        measured : _ -> timing measured
        [] -> Left "Missing batch model load"

timing :: Frame -> Either String ()
timing (Frame raw fields) = void (Duration.admit raw fields)

payload :: Text -> Object -> Either String [ByteString]
payload expected = parseEither $ \value -> do
    Json.fields ["stage", "format", "calls"] value
    actualStage <- value .: "stage"
    actualFormat <- value .: "format"
    unless (actualStage == expected && actualFormat == format) (fail "Unexpected finite inference batch frame")
    calls <- value .: "calls"
    when (null calls) (fail "A finite inference batch must contain calls")
    pure (map encodeUtf8 calls)

member :: [Text] -> ByteString -> Either String [Frame]
member expected encoded = do
    records <- decode encoded
    unless (length records == length expected) (Left "Incomplete or repeated batch member observations")
    mapM_ (uncurry stage) (zip expected records)
    pure records

stage :: Text -> Frame -> Either String ()
stage expected (Frame _ fields) = do
    actual <- parseEither (.: "stage") fields
    unless (actual == expected && not (Fields.member "phase" fields)) (Left "Missing or reordered finite batch observation stage")

takeGroup :: [Frame] -> Either String (Group, [Frame])
takeGroup (first : timed : lastFrame : rest) = do
    prefix <- payload "consumed" (fields first)
    (measured, outputs) <- completion [timed, lastFrame]
    unless (length prefix == length outputs) (Left "Batch completion inventory differs from consumption")
    values <- traverse pair (zip prefix outputs)
    bindings <- traverse (parseEither Wire.binding . fields . consumed) values
    unless (distinct (map Invocation.boundCall bindings) && distinct (map Invocation.boundAttempt bindings) && distinct (map Invocation.boundInstance bindings)) (Left "Batch observation reuses a call, attempt or activation instance")
    pure (Group values measured (encode [first, timed, lastFrame]), rest)
  where
    pair (prefix, output) = do
        inputs <- member ["loaded_adapter", "consumed"] prefix
        outputs <- member ["result"] output
        case (inputs, outputs) of
            ([loaded, consumed], [result]) -> do
                mapM_ (parseEither Wire.binding . fields) [loaded, consumed, result]
                unless (all ((== Fields.lookup "binding" (fields consumed)) . Fields.lookup "binding" . fields) [loaded, result]) (Left "Batch member observations have mismatched bindings")
                pure (Member loaded consumed result)
            _ -> Left "Incomplete batch member observations"
    distinct values = length values == Set.size (Set.fromList values)
takeGroup _ = Left "Incomplete finite batch observations"

groups :: [Frame] -> Either String [Group]
groups [] = pure []
groups records@(first : rest)
    | grouped first = do
        (observed, remaining) <- takeGroup records
        (observed :) <$> groups remaining
    | otherwise = groups rest

memberBytes :: Member -> ByteString
memberBytes value = encode [loaded value, consumed value, result value]
