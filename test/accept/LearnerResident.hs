{-# LANGUAGE GHC2021 #-}

module LearnerResident (execute) where

import Invar.Learn.Worker qualified as Worker
import Invar.Learn.Worker.Resident qualified as Resident

execute :: Resident.Resident owner -> Resident.Paths -> Worker.Call algorithm -> IO (Either Worker.Failure (Resident.Receipt algorithm))
execute = Resident.run
