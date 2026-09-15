{-# LANGUAGE OverloadedStrings #-}

module Invar.Measurement (Source (..), Report, admit, describe) where

import Control.Monad (unless)
import Data.Aeson (ToJSON (..), Value, object, (.=))
import Data.Aeson.Encoding qualified as Encoding
import Data.Aeson.Types (Pair)
import Data.Set qualified as Set
import Data.Text (Text)
import Invar.Evaluation qualified as Evaluation
import Invar.Measurement.Direct qualified as Direct
import Invar.Measurement.Inference qualified as Inference
import Invar.Measurement.Manifest qualified as Manifest
import Invar.Measurement.Run qualified as Run
import Invar.Measurement.Source (Source (..))
import Invar.Measurement.Source qualified as Source
import Invar.Measurement.Statistics qualified as Statistics
import Invar.Workload qualified as Workload

data Report = Report String String String String [Statistics.Summary] Value

admit :: Source -> Workload.Document -> (String, FilePath) -> IO Report
admit source tasks (policy, path) = do
    manifestSnapshot <- Source.snapshot source path
    declared <- Source.checked (Manifest.admit (Source.encoded manifestSnapshot))
    resolved <- traverse (resolvePath source . Manifest.path) (Manifest.runs declared)
    unless (length resolved == Set.size (Set.fromList resolved)) (Source.invalid "Repeated measurement path")
    referenceSnapshot <- Source.snapshot source (Manifest.reference declared)
    reference <- Source.checked (Inference.admit tasks (Evaluation.Run policy (Manifest.referenceExit declared)) (Source.encoded referenceSnapshot))
    let first = Inference.observedRun reference
    summaries <- traverse (observe reference first) (Manifest.runs declared)
    compared <- Source.checked (Statistics.comparison summaries)
    pure (Report (Source.digest manifestSnapshot) (Workload.digest tasks) (Source.digest referenceSnapshot) policy summaries compared)
  where
    observe reference initial run = do
        observed <- case Manifest.route run of
            Manifest.Invar -> do
                snapshot <- Source.snapshot source (Manifest.path run)
                Inference.observedRun <$> Source.checked (Inference.admit tasks (Evaluation.Run policy 0) (Source.encoded snapshot))
            Manifest.Direct -> Direct.admit source (Manifest.path run) (Inference.replayReference reference)
        paired <- Source.checked (Run.paired initial observed)
        Source.checked (Statistics.summarize run paired)

describe :: Report -> Value
describe report@(Report _ _ _ _ runs _) = object (fields report ++ ["runs" .= runs])

instance ToJSON Report where
    toJSON = describe
    toEncoding report@(Report _ _ _ _ runs _) = Encoding.pairs (foldMap (uncurry (.=)) (fields report) <> "runs" .= runs)

fields :: Report -> [Pair]
fields (Report manifest tasks reference policy _ compared) =
    [ "manifest_sha256" .= manifest
    , "tasks_sha256" .= tasks
    , "reference_log_sha256" .= reference
    , "policy" .= policy
    , "comparison" .= compared
    , "scope" .= scope
    ]

scope :: Text
scope = "repeated reported inference measurements with supplied successful exit statuses and elapsed durations; the residual is elapsed time minus the worker critical path (per cohort the longest session when several concurrent physical owners are declared, otherwise the sum, added over cohorts; resident measurements include actual activation, release and sequential final close costs and require matched physical lifetimes); route differences include worker setup, validation, logging and environmental variation, not isolated causal control overhead"
