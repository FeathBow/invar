{-# LANGUAGE GHC2021 #-}

module Logged (forge) where

import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Trajectory qualified as Trajectory

-- Reject: [GHC-01928]
forge :: String -> Trajectory.Trajectory -> Replay.Logged
forge = Replay.Logged
