{-# LANGUAGE GHC2021 #-}

module Update (forge) where

import Invar.Learn.Protocol qualified as P
import Invar.Learn.Worker qualified as W

-- Reject: [GHC-01928]
forge :: P.Result -> W.Execution scope
forge = W.Execution
