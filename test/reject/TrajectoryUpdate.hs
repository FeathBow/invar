{-# LANGUAGE GHC2021 #-}

module TrajectoryUpdate (replace) where

import Invar.Infer.Result qualified as Result
import Invar.Infer.Trajectory qualified as Trajectory

-- Reject: [GHC-22385]
replace :: Result.Result -> Trajectory.Trajectory -> Trajectory.Trajectory
replace observed selected = selected {Trajectory.result = observed}
