{-# LANGUAGE GHC2021 #-}

module InferenceSourceConstructor (forge) where

import Invar.Infer.Observation qualified as Observation
import Invar.Infer.Trajectory qualified as Trajectory

-- Reject: [GHC-01928]
forge :: String -> Trajectory.Trajectory -> Observation.Source
forge = Observation.Source
