{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Artifacts (initial, successor) where

import Control.Monad (unless)
import Data.Aeson (Object, Value, object, withObject, (.:), (.=))
import Data.Aeson.Types (parseEither)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Word (Word32)
import GHC.Float (castDoubleToWord64, castWord32ToFloat, float2Double)
import Invar.History.Cohort qualified as Cohort
import Invar.History.Publication qualified as Publication
import Invar.History.Trace qualified as Trace
import Invar.Json qualified as Json
import Invar.Learn qualified as Learn
import Invar.Learn.Adapter qualified as Adapter
import Invar.Learn.Checkpoint qualified as Checkpoint
import Invar.Learn.Codec qualified as Codec
import Invar.Learn.Gradient qualified as Gradient
import Invar.Learn.Observation qualified as Observation
import Invar.Learn.Report qualified as Report
import Invar.Policy.File qualified as File
import Numeric.Natural (Natural)
import System.FilePath ((</>))
import System.Posix.Files qualified as Posix

initial :: Codec.Decoder -> (Learn.Settings, FilePath) -> IO (Map Text [Integer], Value)
initial decoder (settings, path) = do
    status <- Posix.getSymbolicLinkStatus path
    unless (Posix.isDirectory status) (invalid "Initial checkpoint is not a direct directory")
    schema <- File.withFile (path </> "adapter.safetensors") $ \file -> do
        Adapter.verify (Learn.policy settings) file
        pure (Adapter.schema file)
    parameters <- either invalid pure (Adapter.parameters schema)
    observed <- Codec.withSession decoder $ \session -> do
        decoded <- Codec.decode session (path </> "learner.pt", Learn.learner settings)
        checked <- either invalid pure (Checkpoint.admitInitial settings parameters decoded)
        state session checked
    pure (schema, object ["checkpoint" .= path, "policy" .= Learn.policy settings, "learner" .= Learn.learner settings, "tokenizer" .= Learn.tokenizer settings, "base" .= Learn.base settings, "assembly" .= Learn.assembly settings, "state" .= observed])

successor :: (Codec.Decoder, Map Text [Integer]) -> (Natural, Trace.Generation) -> Publication.Observed -> IO Value
successor (decoder, schema) (index, generation) published = do
    let path = Publication.directory published
        report = Cohort.update (Trace.cohort generation)
    parameters <- either invalid pure (Adapter.parameters schema)
    policy <- either invalid pure (Report.artifact "adapter" report)
    learner <- either invalid pure (Report.artifact "learner" report)
    File.withFile (path </> "adapter.safetensors") $ \file -> Adapter.matches schema file >> Adapter.verify policy file
    observed <- Codec.withSession decoder $ \session -> do
        decoded <- Codec.decode session (path </> "learner.pt", learner)
        checked <- either invalid pure (Checkpoint.admit report parameters decoded)
        steps <- Checkpoint.inspectTensors session checked
        unless (not (null steps) && all (== toInteger index) steps) (invalid "AdamW steps differ from the declared generation")
        stateSummary checked steps
    gradients <- Gradient.observe parameters (report, path </> "gradients.safetensors")
    probabilities <- Observation.probability report (path </> "probabilities.json")
    either invalid pure (roles generation probabilities)
    pure (object ["publication" .= Publication.describe published, "state" .= observed, "gradients" .= gradients, "probabilities" .= probabilities])

state :: Codec.Session -> Checkpoint.Checked -> IO Value
state session checked = Checkpoint.inspectTensors session checked >>= stateSummary checked

stateSummary :: Checkpoint.Checked -> [Integer] -> IO Value
stateSummary checked steps = pure (object ("steps" .= steps : Checkpoint.rngSummary checked))

roles :: Trace.Generation -> [Object] -> Either String ()
roles generation probabilities = do
    named <- traverse (\fields -> (,) <$> parseEither (.: "sample") fields <*> pure fields) probabilities
    mapM_ (check (Map.fromList named)) (Trace.roleOutputs generation)
  where
    check expected encoded = do
        sample <- Json.decode encoded >>= parseEither (withObject "probability role output" (.: "sample"))
        observed <- maybe (Left "Unmatched probability role observation") Right (Map.lookup (sample :: Text) expected)
        mapM_ (vector encoded observed) ["proximal", "reference"]
    vector encoded observed key = do
        actual <- map castDoubleToWord64 <$> Json.floatingArrayAt [key] encoded
        bits <- parseEither (.: key) observed :: Either String [Word32]
        unless (actual == map (castDoubleToWord64 . float2Double . castWord32ToFloat) bits) (Left "Logged probability roles differ from actual artifact words")

invalid :: String -> IO value
invalid = ioError . userError
