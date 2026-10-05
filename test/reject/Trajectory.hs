{-# LANGUAGE GHC2021 #-}

module Trajectory (forge) where

import Invar.Infer.Result qualified as Result
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Spec.Invocation qualified as Invocation

-- Reject: [GHC-01928]
forge :: Invocation.Completion -> Result.Result -> Trajectory.Trajectory
forge = Trajectory.Trajectory
