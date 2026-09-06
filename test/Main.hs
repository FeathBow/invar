module Main (main) where

import Arguments (arguments)
import Control.Monad (unless)
import Fixtures (fixtures)
import Hedgehog (checkSequential)
import Properties (properties)
import System.Exit (exitFailure)

main :: IO ()
main = do
    results <- traverse checkSequential [fixtures, properties, arguments]
    unless (and results) exitFailure
