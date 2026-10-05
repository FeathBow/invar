{-# LANGUAGE GHC2021 #-}

module TrajectoryInternal (forge) where

import Invar.Infer.Trajectory.Internal qualified as Internal

-- Reject: [GHC-87110]
forge :: Internal.Trajectory -> Internal.Trajectory
forge = id
