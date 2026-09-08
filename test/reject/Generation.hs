{-# LANGUAGE GHC2021 #-}

module Generation (forge) where

import Invar.Learn.Worker qualified as W
import Invar.Loop qualified as L

-- Reject: [GHC-01928]
forge :: W.Execution scope -> L.Generation scope
forge = L.Generation
