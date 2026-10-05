{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Framing (Frame (..), stageName, format, decode, encode, grouped, readiness, activationReadiness, completion) where

import Control.Monad (unless, void, when)
import Data.Aeson (Object, Value (..), withObject, (.:))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text (Text)
import Data.Text.Encoding (encodeUtf8)
import Invar.Json qualified as Json
import Invar.Measurement.Duration qualified as Duration

data Frame = Frame {raw :: ByteString, fields :: Object}
    deriving (Eq, Show)

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
