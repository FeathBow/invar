{-# LANGUAGE RoleAnnotations #-}

module Invar.Learn.Worker.Observation (ObservationOf (..), Observation, report) where

import Invar.Cohort qualified as Cohort
import Invar.Learn.Protocol qualified as Protocol
import Invar.Learn.Worker qualified as Worker
import Invar.Learn.Worker.Resident qualified as Resident

type role ObservationOf nominal nominal
data ObservationOf scope grade = Terminated (Worker.ExecutionOf scope grade) | Acknowledged (Resident.ReceiptOf scope grade)

type Observation scope = ObservationOf scope Cohort.Graded

report :: ObservationOf scope grade -> Protocol.Result
report (Terminated value) = Worker.report value
report (Acknowledged value) = Resident.report value
