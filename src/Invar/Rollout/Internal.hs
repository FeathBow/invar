{-# LANGUAGE RoleAnnotations #-}

module Invar.Rollout.Internal (Driver (..), reserve) where

import Control.Concurrent.MVar (MVar)
import Data.IORef (IORef, atomicModifyIORef')
import Numeric.Natural (Natural)

type role Driver nominal
data Driver scope = Driver (MVar ()) (IORef Natural)

reserve :: Driver scope -> Natural -> IO Natural
reserve (Driver _ counter) count = atomicModifyIORef' counter (\next -> (next + count, next))
