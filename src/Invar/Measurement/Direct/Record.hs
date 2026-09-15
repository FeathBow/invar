{-# LANGUAGE OverloadedStrings #-}

module Invar.Measurement.Direct.Record (document, records, positive, checkStderr, padded) where

import Control.Monad (unless, (>=>))
import Data.Aeson (Object, withObject, (.:))
import Data.Aeson.Key (Key)
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString.Char8 qualified as Bytes
import Invar.Json qualified as Json
import Invar.Measurement.Source qualified as Source
import Numeric.Natural (Natural)

document :: Source.Source -> FilePath -> IO (Source.Snapshot, Object)
document source path = do
    snapshot <- Source.snapshot source path
    fields <- Source.checked (Json.decode (Source.encoded snapshot) >>= parseEither (withObject "direct measurement document" pure))
    pure (snapshot, fields)

records :: Bytes.ByteString -> Either String [Object]
records encoded = do
    unless ("\n" `Bytes.isSuffixOf` encoded) (Left "Incomplete measurement record stream")
    traverse (Json.decode >=> parseEither (withObject "direct call record" pure)) (Bytes.lines encoded)

positive :: Key -> Object -> Parser Double
positive key fields = do
    duration <- fields .: key >>= Json.finite
    unless (duration > 0) (fail "Expected a positive measurement duration")
    pure duration

checkStderr :: Source.Source -> FilePath -> Object -> IO ()
checkStderr source path fields = do
    expected <- Source.checked (parseEither (\value -> value .: "stderr_sha256" >>= Json.identity) fields)
    actual <- Source.snapshot source path
    unless (Source.digest actual == expected) (Source.invalid "Direct stderr identity mismatch")

padded :: Natural -> String
padded index = let width = 4; digits = show index in replicate (max 0 (width - length digits)) '0' ++ digits
