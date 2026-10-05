{-# LANGUAGE OverloadedStrings #-}

module Invar.Resident (Owner (..), Role (..), Release, format, ownerValue, prepare, request, retire, released, observeRelease, close, closed, measured) where

import Control.Monad (foldM, unless, void)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value (..), encode, object, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Lazy qualified as Lazy
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text.Encoding (encodeUtf8)
import Invar.Artifact qualified as Artifact
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Measurement.Duration qualified as Duration
import Invar.Spec.Invocation qualified as Invocation
import Invar.Spec.Load qualified as Load
import Numeric.Natural (Natural)

data Role = Inference | Learning | Shared deriving (Eq, Show)
data Owner = Owner Role Natural deriving (Eq, Show)
data Release = Release Owner String [Load.Fact]

format :: Text
format = "invar-resident-v1"

prepare :: Owner -> [Load.Fact] -> ByteString -> Either String Release
prepare owner facts source = do
    let instances = map (Load.instanceName . Load.description) facts
    unless (not (null facts) && length instances == Set.size (Set.fromList instances)) (Left "Resident release requires distinct completed load instances")
    pure (Release owner (Artifact.hex (SHA256.hash source)) facts)

ownerValue :: Owner -> Value
ownerValue (Owner role session) = object ["role" .= name role, "session" .= session]
  where
    name Inference = "inference" :: Text
    name Learning = "learning"
    name Shared = "shared"

releaseValue :: Text -> Release -> Value
releaseValue action (Release owner digest facts) =
    releaseFields (action, owner, digest) (map (\fact -> Wire.invocationValue (Invocation.completedBinding (Load.report fact)) (Invocation.completedProgram (Load.report fact))) facts)

releaseFields :: (Text, Owner, String) -> [Value] -> Value
releaseFields (action, owner, digest) loads =
    object
        [ "format" .= format
        , "owner" .= ownerValue owner
        , "loads" .= loads
        , "result_sha256" .= digest
        , "action" .= action
        ]

request :: Release -> ByteString
request = Lazy.toStrict . encode . releaseValue "release"

released :: Release -> ByteString -> Either String Duration.Duration
released expected = acknowledgement "released" (releaseValue "release" expected)

observeRelease :: (Owner, [Value], ByteString) -> ByteString -> Either String Duration.Duration
observeRelease (owner, loads, source) = acknowledgement "released" (releaseFields ("release", owner, Artifact.hex (SHA256.hash source)) loads)

retire :: Release -> Load.Registry -> ByteString -> Either String Load.Registry
retire expected@(Release _ _ facts) registry encoded = do
    void (released expected encoded)
    let instances = map (Load.instanceName . Load.description) facts
    unless (Set.fromList instances == Set.fromList (Load.active registry)) (Left "Resident release does not cover the complete live activation inventory")
    foldM unload registry facts
  where
    unload current fact = do
        let instanceId = Load.instanceName (Load.description fact)
        original <- first show (Load.historical current instanceId)
        unless (original == fact) (Left "Resident release names a different historical load")
        first show (Load.unload instanceId current)

close :: Owner -> ByteString
close owner = Lazy.toStrict (encode (object ["format" .= format, "owner" .= ownerValue owner, "action" .= ("close" :: Text)]))

closed :: Owner -> Natural -> ByteString -> Either String Duration.Duration
closed owner groups = acknowledgement "closed" (object ["format" .= format, "owner" .= ownerValue owner, "groups" .= groups])

acknowledgement :: Text -> Value -> ByteString -> Either String Duration.Duration
acknowledgement stage expected encoded = do
    fields <- Json.decode encoded >>= parseEither (withObject "resident acknowledgement" pure)
    let actual = Object (Fields.delete "measurement" fields)
        target = case expected of
            Object expectedFields -> Object (Fields.insert "stage" (String stage) (Fields.delete "action" expectedFields))
            _ -> expected
    unless (actual == target) (Left "Resident acknowledgement differs from its owner or exact completed inventory")
    measured stage encoded

measured :: Text -> ByteString -> Either String Duration.Duration
measured stage encoded = do
    fields <- Json.decode encoded >>= parseEither (withObject "resident acknowledgement" pure)
    operation <- encodeUtf8 <$> parseEither (.: "measurement") fields
    frames <- Framing.decode operation
    case frames of
        [Framing.Frame raw values] -> do
            unless (Fields.lookup "stage" values == Just (String stage)) (Left "Resident acknowledgement has a different measured operation")
            Duration.admit raw values
        _ -> Left "Resident acknowledgement requires one original measured operation"
