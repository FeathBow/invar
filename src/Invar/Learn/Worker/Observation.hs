{-# LANGUAGE RoleAnnotations #-}

module Invar.Learn.Worker.Observation (Observation (..), report, plan) where

import Invar.Learn qualified as Learn
import Invar.Learn.Protocol qualified as Protocol
import Invar.Learn.Worker qualified as Worker
import Invar.Learn.Worker.Resident qualified as Resident

type role Observation nominal
data Observation scope = Terminated (Worker.Execution scope) | Acknowledged (Resident.Receipt scope)

report :: Observation scope -> Protocol.Result
report (Terminated value) = Worker.report value
report (Acknowledged value) = Resident.report value

plan :: Observation scope -> Learn.Plan scope
plan (Terminated value) = Worker.plan value
plan (Acknowledged value) = Resident.plan value
