{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Observation (Input (..), report, gradients, probabilities, probability) where

import Control.Monad (unless)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Object, Value, object, withObject, (.:), (.=))
import Data.Aeson.Key qualified as Key
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString qualified as Bytes
import Data.Text (Text)
import Invar.Artifact qualified as Artifact
import Invar.Learn.Gradient qualified as Gradient
import Invar.Learn.Probability qualified as Probability
import Invar.Learn.Report qualified as Report
import Numeric.Natural (Natural)

data Input = Input {artifact :: FilePath, logPath :: FilePath, call :: Natural}

report :: Input -> IO Report.Report
report input = Bytes.readFile (logPath input) >>= either invalid pure . Report.admit (call input)

gradients :: FilePath -> Input -> Input -> IO Value
gradients policy left right = do
    first <- report left
    second <- report right
    Gradient.compare policy (first, artifact left) (second, artifact right)

probability :: Report.Report -> FilePath -> IO [Object]
probability expected path = do
    encoded <- Bytes.readFile path
    digest <- either invalid pure (Report.artifact "probabilities" expected)
    unless (Artifact.hex (SHA256.hash encoded) == digest) (invalid "Probability file differs from its reported digest")
    either invalid pure (Probability.observe (Report.invocation expected, Report.request expected, Report.output expected) encoded)

probabilities :: Input -> Input -> IO Value
probabilities left right = do
    first <- report left
    second <- report right
    either invalid pure (Report.paired first second)
    initial <- probability first (artifact left)
    changed <- probability second (artifact right)
    differences <- traverse difference [(old, new) | (old, new) <- zip initial changed, not (null (fields old new))]
    leftBinding <- binding first
    rightBinding <- binding second
    pure (object ["comparison" .= ("pre-AdamW objective input and scalar words" :: Text), "equal" .= null differences, "left_binding" .= leftBinding, "right_binding" .= rightBinding, "samples" .= length initial, "differences" .= differences])
  where
    binding = either invalid pure . parseEither (withObject "invocation" (.: "binding")) . Report.invocation :: Report.Report -> IO Value
    fields old new = [Key.toText role | role <- ["behavior", "proximal", "reference", "current", "advantage", "objective"], Fields.lookup role old /= Fields.lookup role new]
    difference (old, new) = do
        name <- either invalid pure (parseEither (.: "sample") old) :: IO Text
        pure (object ["sample" .= name, "fields" .= fields old new])

invalid :: String -> IO value
invalid = ioError . userError
