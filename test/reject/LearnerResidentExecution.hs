{-# LANGUAGE GHC2021 #-}

module LearnerResidentExecution (promote) where

import Invar.Learn.Worker qualified as Worker
import Invar.Learn.Worker.Resident qualified as Resident

-- Reject: Couldn't match type
promote :: Resident.Receipt scope -> Worker.Execution scope
promote = id
