{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Artifacts (initial, successor) where

import Control.Monad (unless)
import Data.Aeson (Value, object, withObject, (.=))
import Data.Aeson.Types (parseEither)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Invar.History.Cohort qualified as Cohort
import Invar.History.Generation qualified as Generation
import Invar.History.Publication qualified as Publication
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
import Invar.Learn.Step qualified as Step
import Invar.Learn.Stream qualified as S
import System.FilePath ((</>))
import System.Posix.Files qualified as Posix

initial :: Codec.Decoder -> (Learn.Settings, FilePath) -> IO (Map Text [Integer], Value, State.Initial)
initial decoder (settings, path) = do
    status <- Posix.getSymbolicLinkStatus path
    unless (Posix.isDirectory status) (invalid "Initial checkpoint is not a direct directory")
    observed <- Codec.withSession decoder $ \session -> State.observeInitial session (settings, path)
    summary <- stateSummary (State.initialChecked observed) (State.initialSteps observed)
    pure (State.initialSchema observed, object ["checkpoint" .= path, "policy" .= Learn.policy settings, "learner" .= Learn.learner settings, "tokenizer" .= Learn.tokenizer settings, "base" .= Learn.base settings, "assembly" .= Learn.assembly settings, "state" .= summary], observed)

successor :: (Codec.Decoder, Map Text [Integer]) -> (Integer, Generation.Generation) -> Publication.Observed -> IO (Value, State.Observed, Gradient.Observed)
successor (decoder, schema) (expected, generation) published = do
    let path = Publication.directory published
        report = Cohort.update (Generation.cohort generation)
    parameters <- either invalid pure (Adapter.parameters schema)
    state' <- Codec.withSession decoder $ \session -> State.observe session (report, schema, path)
    let counted = State.steps state'
    unless (not (null counted) && all (== expected) counted) (invalid "AdamW steps differ from the optimizer steps of the committed updates")
    observed <- stateSummary (State.checked state') counted
    (gradients, gradient) <- Gradient.observe parameters (report, path </> "gradients.safetensors")
    probabilities <- Observation.probability report (path </> "probabilities.json")
    either invalid pure (reported generation probabilities)
    mismatch <- either invalid pure (Mismatch.summarize [(Probability.behavior sample, Probability.proximal sample) | sample <- probabilities])
    compared <- case [(Probability.engineReference sample, learned) | sample <- probabilities, Just learned <- [Probability.reference sample]] of
        [] -> pure Nothing
        pairs -> Just <$> either invalid pure (Mismatch.summarize pairs)
    let diagnostics = ["reference_engine" .= Mismatch.describe value | Just value <- [compared]]
    pure (object (["publication" .= Publication.describe published, "state" .= observed, "gradients" .= gradients, "probabilities" .= map Probability.sampleObject probabilities, "learner_engine" .= Mismatch.describe mismatch] ++ diagnostics), state', gradient)

stateSummary :: Checkpoint.Checked -> [Integer] -> IO Value
stateSummary checked steps = pure (object ("steps" .= steps : Checkpoint.rngSummary checked))

reported :: Generation.Generation -> [Probability.Sample] -> Either String ()
reported generation probabilities = mapM_ check (Generation.stepOutputs generation)
  where
    expected = Map.fromList [(Probability.sampleName sample, sample) | sample <- probabilities]
    check encoded = do
        decoded <- Json.decode encoded >>= parseEither (withObject "learner step record" Step.decode)
        case decoded of
            Just (Step.Proximal name words32) -> matches name words32 (Just . Probability.proximal)
            Just (Step.Reference name words32) -> matches name words32 Probability.reference
            Just (Step.Current report) -> matches (S.sample report) (S.words32 report) (lookup (S.step report) . Probability.currents)
            _ -> Left "A logged learner record does not report step words"
    matches name words32 role = do
        observed <- maybe (Left "Unmatched learner step record") Right (Map.lookup name expected)
        unless (role observed == Just words32) (Left "Logged learner step words differ from the probability artifact")

invalid :: String -> IO value
invalid = ioError . userError
