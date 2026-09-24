{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Artifacts (initial, successor) where

import Control.Monad (unless)
import Data.Aeson (Value, object, withObject, (.:), (.=))
import Data.Aeson.Types (parseEither)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import GHC.Float (castDoubleToWord64)
import Invar.Float32 qualified as Float32
import Invar.History.Cohort qualified as Cohort
import Invar.History.Publication qualified as Publication
import Invar.History.Trace qualified as Trace
import Invar.Json qualified as Json
import Invar.Learn qualified as Learn
import Invar.Learn.Adapter qualified as Adapter
import Invar.Learn.Checkpoint qualified as Checkpoint
import Invar.Learn.Codec qualified as Codec
import Invar.Learn.Gradient qualified as Gradient
import Invar.Learn.Mismatch qualified as Mismatch
import Invar.Learn.Observation qualified as Observation
import Invar.Learn.Probability qualified as Probability
import Invar.Learn.State qualified as State
import Numeric.Natural (Natural)
import System.FilePath ((</>))
import System.Posix.Files qualified as Posix

initial :: Codec.Decoder -> (Learn.Settings, FilePath) -> IO (Map Text [Integer], Value, State.Initial)
initial decoder (settings, path) = do
    status <- Posix.getSymbolicLinkStatus path
    unless (Posix.isDirectory status) (invalid "Initial checkpoint is not a direct directory")
    observed <- Codec.withSession decoder $ \session -> State.observeInitial session (settings, path)
    summary <- stateSummary (State.initialChecked observed) (State.initialSteps observed)
    pure (State.initialSchema observed, object ["checkpoint" .= path, "policy" .= Learn.policy settings, "learner" .= Learn.learner settings, "tokenizer" .= Learn.tokenizer settings, "base" .= Learn.base settings, "assembly" .= Learn.assembly settings, "state" .= summary], observed)

successor :: (Codec.Decoder, Map Text [Integer]) -> (Natural, Trace.Generation) -> Publication.Observed -> IO (Value, State.Observed, Gradient.Observed)
successor (decoder, schema) (index, generation) published = do
    let path = Publication.directory published
        report = Cohort.update (Trace.cohort generation)
    parameters <- either invalid pure (Adapter.parameters schema)
    state' <- Codec.withSession decoder $ \session -> State.observe session (report, schema, path)
    let counted = State.steps state'
    unless (not (null counted) && all (== toInteger index) counted) (invalid "AdamW steps differ from the declared generation")
    observed <- stateSummary (State.checked state') counted
    (gradients, gradient) <- Gradient.observe parameters (report, path </> "gradients.safetensors")
    probabilities <- Observation.probability report (path </> "probabilities.json")
    either invalid pure (roles generation probabilities)
    mismatch <- either invalid pure (Mismatch.summarize [(Probability.behavior sample, Probability.proximal sample) | sample <- probabilities])
    pure (object ["publication" .= Publication.describe published, "state" .= observed, "gradients" .= gradients, "probabilities" .= map Probability.sampleObject probabilities, "learner_engine" .= Mismatch.describe mismatch], state', gradient)

stateSummary :: Checkpoint.Checked -> [Integer] -> IO Value
stateSummary checked steps = pure (object ("steps" .= steps : Checkpoint.rngSummary checked))

roles :: Trace.Generation -> [Probability.Sample] -> Either String ()
roles generation probabilities = mapM_ check (Trace.roleOutputs generation)
  where
    expected = Map.fromList [(Probability.sampleName sample, sample) | sample <- probabilities]
    check encoded = do
        (decoded, arrays) <- Json.decodeWithArrays [["proximal"], ["reference"]] encoded
        (proximal, fixed) <- case arrays of
            [first, second] -> Right (first, second)
            _ -> Left "Expected the proximal and reference arrays"
        name <- parseEither (withObject "probability role output" (.: "sample")) decoded
        observed <- maybe (Left "Unmatched probability role observation") Right (Map.lookup (name :: Text) expected)
        unless (map castDoubleToWord64 proximal == map Float32.widened (Probability.proximal observed) && map castDoubleToWord64 fixed == map Float32.widened (Probability.fixed observed)) (Left "Logged probability roles differ from actual artifact words")

invalid :: String -> IO value
invalid = ioError . userError
