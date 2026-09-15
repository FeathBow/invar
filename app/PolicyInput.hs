module PolicyInput (run, plan, selection) where

import Control.Monad (unless, when)
import Data.ByteString qualified as Bytes
import Data.Maybe (isJust)
import InferenceInput qualified
import Invar.Infer qualified as Infer
import Invar.Infer.Observation qualified as Observation
import Invar.Policy qualified as Policy
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)
import System.FilePath ((</>))

run :: [String] -> IO ()
run ["inspect", "--help"] = putStrLn (usageInfo "Usage: invar policy inspect --checkpoint DIRECTORY\nRead the strict policy description and verify its canonical adapter contents. This grants no loading or numerical authority." inspectionOptions)
run ("inspect" : supplied) = do
    fields <- either die pure (O.parse inspectionOptions supplied)
    checkpoint <- either die pure (O.required fields "checkpoint")
    selected <- Policy.readDescription (checkpoint </> "policy.json")
    checkAdapter checkpoint selected
    Bytes.putStr (Policy.encodeDescription selected)
run ["--help"] = putStrLn (usageInfo "Usage: invar policy OPTIONS\nSeal checkpoint/policy.json from an admitted standalone inference observation. The selected checkpoint adapter must match the observation. This stages an initial description; it does not issue a publication receipt or numerical certificate." options)
run supplied = do
    fields <- either die pure (O.parse options supplied)
    requested <- either die pure (InferenceInput.request fields)
    planned <- either (die . show) pure (Infer.prepare requested)
    bound <- either die pure (InferenceInput.binding fields)
    path <- either die pure (O.required fields "log")
    status <- either die pure (O.numeric fields "exit-code")
    unless (status == (0 :: Int)) (die "Inference process did not exit successfully")
    observed <- Bytes.readFile path >>= either die pure . Observation.admit planned bound
    selected <- either die pure (Observation.policyDescription observed)
    checkpoint <- either die pure (O.required fields "checkpoint")
    checkAdapter checkpoint selected
    Policy.stageDescription (checkpoint </> "policy.json") selected
    Bytes.putStr (Policy.encodeDescription selected)

checkAdapter :: FilePath -> Policy.Description -> IO ()
checkAdapter checkpoint selected = do
    actual <- Policy.identity (checkpoint </> "adapter.safetensors")
    unless (actual == Policy.adapter selected) (die "Checkpoint adapter differs from the selected inference policy")

plan :: O.Fields -> IO (FilePath, Infer.Plan)
plan fields = do
    (adapter, selected) <- selection "digest" fields
    requested <- either die pure (maybe InferenceInput.request InferenceInput.requestFor selected fields)
    planned <- either (die . show) pure (Infer.prepare requested >>= maybe Right Infer.bindPolicy selected)
    pure (adapter, planned)

selection :: String -> O.Fields -> IO (FilePath, Maybe Policy.Description)
selection identityOption fields = case O.optional fields "checkpoint" of
    Nothing -> (,Nothing) <$> either die pure (O.required fields "adapter")
    Just checkpoint -> do
        when (any (isJust . O.optional fields) ["adapter", identityOption, "tokenizer-digest", "base-digest", "assembly-digest"]) (die "--checkpoint derives adapter and materialization bindings from policy.json; separate adapter/digest options are invalid")
        selected <- Policy.readDescription (checkpoint </> "policy.json")
        pure (checkpoint </> "adapter.safetensors", Just selected)

options :: [OptDescr (String, String)]
options = InferenceInput.options ++ O.descriptions [("checkpoint", "Initial checkpoint whose adapter matches the observed inference"), ("log", "Complete standalone inference stdout"), ("exit-code", "Independently observed inference process exit status")]

inspectionOptions :: [OptDescr (String, String)]
inspectionOptions = O.descriptions [("checkpoint", "Checkpoint containing policy.json and adapter.safetensors")]
