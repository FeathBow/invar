{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Report.Internal (Report (..), admitted) where

import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Object, Value, object, (.:), (.=))
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.Text (Text)
import Invar.Artifact qualified as Artifact
import Invar.Json qualified as Json
import Invar.Learn.Request qualified as Request

data Report = Report String Value Request.Request (Object, ByteString) String

admitted :: ByteString -> (Value, Text, Request.Request) -> (Object, ByteString) -> Either String Report
admitted source (bound, program, input) (returned, encoded) = do
    identity <- parseEither (\fields -> fields .: "gradients" >>= Json.identity) returned
    pure (Report (Artifact.hex (SHA256.hash source)) (object ["binding" .= bound, "program" .= program]) input (returned, encoded) identity)
