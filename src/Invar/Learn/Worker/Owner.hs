module Invar.Learn.Worker.Owner (Runner, withRecordedRunner, borrowed, run) where

import Invar.Learn.Worker qualified as Worker
import Invar.Learn.Worker.Observation qualified as Observation
import Invar.Learn.Worker.Resident qualified as Resident
import Invar.Transcript qualified as Transcript

newtype Runner = Runner {run :: forall kind (scope :: kind) grade. Resident.Paths -> Worker.CallOf scope grade -> IO (Either Worker.Failure (Observation.ObservationOf scope grade))}

withRecordedRunner :: Worker.Mode -> Worker.Worker -> IO Transcript.Transcript -> (Runner -> IO value) -> IO (Either Worker.Failure value)
withRecordedRunner Worker.Process worker opened action = Right <$> action (Runner execute)
  where
    execute paths call = do
        transcript <- opened
        fmap Observation.Terminated <$> Worker.run worker {Worker.checkpoint = Resident.checkpoint paths, Worker.output = Resident.output paths} transcript call
withRecordedRunner Worker.Resident worker opened action = do
    transcript <- opened
    Resident.withResident (Resident.Options worker learnerOwner transcript) $ \owner ->
        Right <$> action (Runner (\paths call -> fmap Observation.Acknowledged <$> Resident.run owner paths call))
  where
    learnerOwner = 0
withRecordedRunner Worker.Shared _ _ _ = pure (Left (Worker.ProtocolFailure "Shared learning requires a joint inference and learning owner"))

borrowed :: Resident.Resident owner -> Runner
borrowed owner = Runner (\paths call -> fmap Observation.Acknowledged <$> Resident.run owner paths call)
