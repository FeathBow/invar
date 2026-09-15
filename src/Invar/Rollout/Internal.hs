{-# LANGUAGE RoleAnnotations #-}

module Invar.Rollout.Internal (Driver (..), reserve) where

import Control.Concurrent.MVar (MVar)
import Data.IORef (IORef, atomicModifyIORef')
import Invar.Rollout.Resident qualified as Resident
import Numeric.Natural (Natural)

type role Driver nominal
data Driver scope = Driver (MVar ()) (IORef Natural) (Maybe Resident.Pool)

reserve :: Driver scope -> Natural -> IO Natural
reserve (Driver _ counter _) count = atomicModifyIORef' counter (\next -> (next + count, next))
