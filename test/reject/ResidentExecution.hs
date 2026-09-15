{-# LANGUAGE GHC2021 #-}

module ResidentExecution (promote) where

import Invar.Worker qualified as Worker
import Invar.Worker.Resident qualified as Resident

-- Reject: Couldn't match type
promote :: Resident.Receipt -> Worker.Execution
promote = id
