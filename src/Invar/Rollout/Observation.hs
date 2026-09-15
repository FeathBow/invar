module Invar.Rollout.Observation (Observation (..), report, completion, loaded, qualified) where

import Invar.Infer.Result qualified as Result
import Invar.Spec.Invocation qualified as Invocation
import Invar.Spec.Load qualified as Load
import Invar.Spec.Qualification qualified as Qualification
import Invar.Worker qualified as Worker
import Invar.Worker.Resident qualified as Resident

data Observation = Terminated Worker.Execution | Acknowledged Resident.Receipt

report :: Observation -> Result.Result
report (Terminated value) = Worker.report value
report (Acknowledged value) = Resident.report value

completion :: Observation -> Invocation.Completion
completion (Terminated value) = Worker.completion value
completion (Acknowledged value) = Resident.completion value

loaded :: Observation -> Load.Fact
loaded (Terminated value) = Worker.loaded value
loaded (Acknowledged value) = Resident.loaded value

qualified :: Observation -> Maybe Qualification.QualifiedResult
qualified (Terminated value) = Worker.qualified value
qualified (Acknowledged value) = Resident.qualified value
