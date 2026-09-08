module Main (main) where

import Arguments (arguments)
import Artifacts (artifacts)
import Collections (collections)
import Construction (construction)
import Control.Monad (unless)
import Dependencies (dependencies)
import Evaluation (evaluation)
import Evidence (evidence)
import Fixtures (fixtures)
import Hedgehog (checkSequential)
import Invocations (invocations)
import Literals (literals)
import Loads (loads)
import Properties (properties)
import Records (records)
import System.Exit (exitFailure)
import Values (values)

main :: IO ()
main = do
    outcomes <- traverse checkSequential [fixtures, properties, arguments, values, dependencies, evaluation, artifacts, construction, collections, records, literals, invocations, evidence, loads]
    unless (and outcomes) exitFailure
