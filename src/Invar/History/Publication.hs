{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Publication (Observed, observe, source, directory, describe) where

import Control.Monad (unless)
import Data.Aeson (Value, object, withObject, (.:), (.=))
import Data.Aeson.Types (Parser, parseEither)
import Invar.Artifact qualified as Artifact
import Invar.History.Cohort qualified as Cohort
import Invar.History.Trace qualified as Trace
import Invar.Infer.Observation qualified as Inference
import Invar.Learn.Report qualified as Report
import Invar.Policy qualified as Policy
import System.FilePath (isRelative, takeDirectory, takeFileName, (</>))
import System.Posix.Files qualified as File

data Observed = Observed FilePath Value

observe :: Trace.Generation -> IO Observed
observe generation = do
    (path, method) <- either invalid pure (parseEither location (Trace.publication generation))
    parent <- File.getSymbolicLinkStatus (takeDirectory path)
    unless (File.isDirectory parent) (invalid "Publication output is not a direct directory")
    entry <- File.getSymbolicLinkStatus path
    target <- case method of
        "rename" -> do
            unless (File.isDirectory entry) (invalid "Rename publication is not a direct directory")
            pure path
        "reference" -> do
            unless (File.isSymbolicLink entry) (invalid "Reference publication is not a symbolic link")
            relative <- File.readSymbolicLink path
            unless (isRelative relative && relative == takeFileName relative && relative `notElem` ["", ".", ".."]) (invalid "Publication reference must name one relative path component")
            let retained = takeDirectory path </> relative
            backing <- File.getSymbolicLinkStatus retained
            unless (File.isDirectory backing) (invalid "Publication reference has no direct retained directory")
            actual <- File.getFileStatus path
            unless (same actual backing) (invalid "Publication reference differs from its retained directory")
            pure retained
        _ -> invalid "Unknown publication method"
    mapM_ (member path target) ["adapter.safetensors", "learner.pt", "policy.json", "gradients.safetensors", "probabilities.json"]
    let report = Cohort.update (Trace.cohort generation)
    policy <- either invalid pure (Report.artifact "adapter" report)
    learner <- either invalid pure (Report.artifact "learner" report)
    actualPolicy <- Policy.identity (path </> "adapter.safetensors")
    actualLearner <- Artifact.identity "Published learner" (path </> "learner.pt")
    unless (actualPolicy == policy && actualLearner == learner) (invalid "Published checkpoint contents differ from the reported update")
    selected <- Policy.readDescription (path </> "policy.json")
    expected <- either invalid pure (source generation >>= Policy.successor policy)
    unless (selected == expected) (invalid "Published policy description differs from the consumed behavior model and updated adapter")
    pure (Observed path (object ["checkpoint" .= path, "publication" .= method, "retained" .= target, "policy" .= policy, "learner" .= learner]))
  where
    location = withObject "checked publication" $ \fields -> (,) <$> fields .: "checkpoint" <*> (fields .: "publication" :: Parser String)

source :: Trace.Generation -> Either String Policy.Description
source generation = do
    descriptions <- traverse Inference.policyDescription (Cohort.inferences (Trace.cohort generation))
    case descriptions of
        initial : remaining -> do
            unless (all (== initial) remaining) (Left "A cohort loaded different inference policy descriptions")
            pure initial
        [] -> Left "Publication has no observed inference policy source"

member :: FilePath -> FilePath -> FilePath -> IO ()
member published retained name = do
    actual <- File.getSymbolicLinkStatus (published </> name)
    backing <- File.getSymbolicLinkStatus (retained </> name)
    unless (File.isRegularFile actual && File.isRegularFile backing && same actual backing) (invalid "Published artifact is not the retained regular file")

same :: File.FileStatus -> File.FileStatus -> Bool
same first second = File.deviceID first == File.deviceID second && File.fileID first == File.fileID second

directory :: Observed -> FilePath
directory (Observed path _) = path

describe :: Observed -> Value
describe (Observed _ value) = value

invalid :: String -> IO value
invalid = ioError . userError
