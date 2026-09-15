{-# LANGUAGE GHC2021 #-}

module LearnerReceipt (forge) where

import Invar.Learn qualified as Learn
import Invar.Learn.Protocol qualified as Protocol
import Invar.Learn.Worker.Resident qualified as Resident

-- Reject: [GHC-01928]
forge :: Learn.Plan scope -> Protocol.Result -> Resident.Receipt scope
forge = Resident.Receipt
