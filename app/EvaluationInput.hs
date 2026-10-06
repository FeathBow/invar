module EvaluationInput (declared, options) where

import Control.Monad (when)
import Data.Maybe (isJust)
import Invar.Evaluation qualified as Evaluation
import Invar.Policy qualified as Policy
import Options qualified as O
import System.Console.GetOpt (OptDescr)
import System.Exit (die)
import System.FilePath ((</>))

declared :: String -> O.Fields -> IO Evaluation.Run
declared prefix fields = do
    status <- either (die . ("Evaluation process: " ++)) pure (O.numeric fields (prefix ++ "exit-code"))
    mode <- case O.optional fields (prefix ++ "worker-mode") of
        Nothing -> pure Evaluation.Serial
        Just "serial" -> pure Evaluation.Serial
        Just "batch" -> pure Evaluation.Batched
        Just "resident" -> pure Evaluation.Resident
        Just _ -> die "An evaluation runs serial, batch or resident"
    case O.optional fields (prefix ++ "checkpoint") of
        Just checkpoint -> do
            when (isJust (O.optional fields (prefix ++ "policy"))) (die ("--" ++ prefix ++ "checkpoint derives the policy identity from policy.json; --" ++ prefix ++ "policy is invalid with it"))
            selected <- Policy.readDescription (checkpoint </> "policy.json")
            pure (Evaluation.Run (Policy.adapter selected) status mode (Just selected))
        Nothing -> do
            policy <- either die pure (O.required fields (prefix ++ "policy"))
            pure (Evaluation.Run policy status mode Nothing)

options :: String -> [OptDescr (String, String)]
options prefix = O.descriptions [(prefix ++ "log", "Complete invar evaluate stdout"), (prefix ++ "policy", "Expected canonical adapter identity"), (prefix ++ "checkpoint", "Checkpoint whose policy.json the evaluation bound, in place of the policy identity"), (prefix ++ "worker-mode", "Worker mode the evaluation ran with: serial (default), batch or resident"), (prefix ++ "exit-code", "Independently observed evaluation process exit status")]
