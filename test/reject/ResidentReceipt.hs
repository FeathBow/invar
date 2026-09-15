{-# LANGUAGE GHC2021 #-}

module ResidentReceipt (forge) where

import Invar.Infer.Result qualified as Result
import Invar.Spec.Invocation qualified as Invocation
import Invar.Worker.Resident qualified as Resident

-- Reject: [GHC-01928]
forge :: Invocation.Completion -> Result.Result -> Resident.Receipt
forge = Resident.Receipt
