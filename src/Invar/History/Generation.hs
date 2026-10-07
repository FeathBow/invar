{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Generation (Generation, Publication (..), generation, cohort, publication, diagnostics, stepOutputs, profiles, describe) where

import Control.Monad (unless)
import Data.Aeson (Value (..), object, toJSON, (.=))
import Data.ByteString (ByteString)
import Invar.Cohort qualified as C
import Invar.History.Cohort qualified as Cohort
import Invar.History.Profile qualified as Profile
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Replay qualified as Replay
import Invar.Learn qualified as Learn
import Invar.Learn.Report qualified as Report
import Invar.Learn.Request qualified as Request
import Invar.Learn.Step qualified as Step
import Invar.Learn.Trace qualified as Trace
import Invar.Resident.Group qualified as Group
import System.FilePath ((</>))

data Publication = Publication {directory :: FilePath, method :: String}
    deriving (Eq, Show)

data Generation = Generation Cohort.Checked Publication [Framing.Frame] [Framing.Frame] [Group.Group] [Profile.Observation]

generation :: Learn.Settings -> (FilePath, String) -> ([C.Task], [Replay.Logged]) -> Trace.Attempt -> ([Framing.Frame], [Group.Group], [Profile.Observation]) -> Either String Generation
generation settings (output, declared) inputs attempted (context, groups, observations) = do
    unless (declared `elem` ["rename", "reference"]) (Left "Unknown declared publication method")
    reported <- maybe (Left ("A generation needs a learner attempt with an admitted result" ++ maybe "" (": " ++) (Trace.stopped attempted))) Right (Trace.result attempted)
    observed <- Cohort.admit settings inputs reported
    let version = Request.scheduled (Report.checkedRequest reported) + 1
    pure (Generation observed (Publication (output </> ("generation" ++ show version)) declared) (Trace.records attempted) context groups observations)

cohort :: Generation -> Cohort.Checked
cohort (Generation observed _ _ _ _ _) = observed

publication :: Generation -> Publication
publication (Generation _ published _ _ _ _) = published

diagnostics :: Generation -> [Value]
diagnostics (Generation _ _ learned context _ _) = map (Object . Framing.fields) (learned ++ context)

stepOutputs :: Generation -> [ByteString]
stepOutputs (Generation _ _ learned _ _ _) = [Framing.raw record | record <- learned, Framing.stageName record `elem` map (Just . String) Step.reports]

profiles :: Generation -> [Profile.Observation]
profiles (Generation _ _ _ _ _ observations) = observations

describe :: Generation -> Value
describe (Generation observed published learned context groups _) =
    object
        [ "cohort" .= Cohort.describe observed
        , "publication" .= object ["checkpoint" .= directory published, "publication" .= method published, "policy" .= identity "adapter", "learner" .= identity "learner"]
        , "diagnostics" .= map (Object . Framing.fields) (learned ++ context)
        , "resident_groups" .= map Group.describe groups
        ]
  where
    identity name = either (const Null) toJSON (Report.artifact name (Cohort.update observed))
