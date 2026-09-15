{-# LANGUAGE GHC2021 #-}

module WorkloadConstructor (forge) where

import Data.Aeson (Value (Null))
import Invar.Workload qualified as Workload

-- Reject: [GHC-01928]
forge :: Workload.Document
forge = Workload.Document "input" Null []
