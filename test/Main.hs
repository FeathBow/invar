module Main (main) where

import Arguments (arguments)
import Artifacts (artifacts)
import BatchCalls (batchCalls)
import Calls (calls)
import Checkpoints (checkpoints)
import Collections (collections)
import Construction (construction)
import Control.Monad (unless)
import Dependencies (dependencies)
import Evaluation (evaluation)
import Evidence (evidence)
import Fixtures (fixtures)
import Hedgehog (checkSequential)
import Inference (inference)
import Invocations (invocations)
import Literals (literals)
import Loads (loads)
import Properties (properties)
import Records (records)
import Results (results)
import Store (store)
import System.Exit (exitFailure)
import Values (values)

main :: IO ()
main = do
    outcomes <- traverse checkSequential [fixtures, properties, arguments, values, dependencies, evaluation, artifacts, construction, collections, records, literals, invocations, evidence, loads, store, checkpoints, inference, results, calls, batchCalls]
    unless (and outcomes) exitFailure
