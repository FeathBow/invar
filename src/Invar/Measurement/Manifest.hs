{-# LANGUAGE OverloadedStrings #-}

module Invar.Measurement.Manifest (Manifest, Run, Route (..), admit, reference, referenceExit, runs, name, route, path, elapsed, routeValue) where

import Control.Monad (unless, when)
import Data.Aeson (Object, Value (..), withObject, (.:))
import Data.Aeson.Key (Key)
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import Invar.Json qualified as Json

data Route = Invar | Direct deriving (Eq, Ord, Show)
data Run = Run {name :: Text, route :: Route, path :: FilePath, elapsed :: Double}
data Manifest = Manifest {reference :: FilePath, referenceExit :: Int, runs :: [Run]}

admit :: ByteString -> Either String Manifest
admit encoded = Json.decode encoded >>= parseEither (withObject "measurement manifest" parse)
  where
    parse fields = do
        Json.fields ["reference_log", "reference_exit_code", "runs"] fields
        logPath <- fields .: "reference_log" >>= nonempty
        status <- successful "reference_exit_code" fields
        supplied <- fields .: "runs" >>= traverse (withObject "measurement run" parseRun)
        when (null supplied) (fail "Missing performance runs")
        unless (length supplied == Set.size (Set.fromList (map name supplied))) (fail "Repeated measurement identity")
        pure (Manifest (Text.unpack logPath) status supplied)

parseRun :: Object -> Parser Run
parseRun fields = do
    Json.fields ["name", "route", "path", "exit_code", "elapsed_seconds"] fields
    identifier <- fields .: "name" >>= nonempty
    location <- fields .: "path" >>= nonempty
    _ <- successful "exit_code" fields
    selected <- fields .: "route" :: Parser Text
    observedRoute <- case selected of
        "invar" -> pure Invar
        "direct" -> pure Direct
        _ -> fail "Unknown measurement route"
    duration <- fields .: "elapsed_seconds" >>= Json.finite
    unless (duration > 0) (fail "Expected a positive measurement duration")
    pure (Run identifier observedRoute (Text.unpack location) duration)

successful :: Key -> Object -> Parser Int
successful key fields = do
    status <- fields .: key
    unless (status == 0) (fail "Measured process did not exit successfully")
    pure status

nonempty :: Text -> Parser Text
nonempty text = do
    when (Text.null text || Text.any (== '\0') text) (fail "Missing or invalid run identity or path")
    pure text

routeValue :: Route -> Value
routeValue Invar = String "invar"
routeValue Direct = String "direct"
