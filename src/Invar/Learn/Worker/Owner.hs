module Invar.Learn.Worker.Owner (Runner, withRunner, withRecordedRunner, borrowed, run) where

import Invar.Learn.Worker qualified as Worker
import Invar.Learn.Worker.Observation qualified as Observation
import Invar.Learn.Worker.Resident qualified as Resident
import Invar.Transcript qualified as Transcript

newtype Runner = Runner {run :: forall kind (scope :: kind). Resident.Paths -> Worker.Call scope -> IO (Either Worker.Failure (Observation.Observation scope))}

withRunner :: Worker.Mode -> Worker.Worker -> (Runner -> IO value) -> IO (Either Worker.Failure value)
withRunner mode worker = runner mode worker (pure Transcript.standard)

withRecordedRunner :: Worker.Mode -> Worker.Worker -> IO Transcript.Transcript -> (Runner -> IO value) -> IO (Either Worker.Failure value)
withRecordedRunner = runner

runner :: Worker.Mode -> Worker.Worker -> IO Transcript.Transcript -> (Runner -> IO value) -> IO (Either Worker.Failure value)
runner Worker.Process worker opened action = Right <$> action (Runner execute)
  where
    execute paths call = do
        transcript <- opened
        fmap Observation.Terminated <$> Worker.run worker {Worker.checkpoint = Resident.checkpoint paths, Worker.output = Resident.output paths} transcript call
runner Worker.Resident worker opened action = do
    transcript <- opened
    Resident.withResident (Resident.Options worker learnerOwner transcript) $ \owner ->
        Right <$> action (Runner (\paths call -> fmap Observation.Acknowledged <$> Resident.run owner paths call))
  where
    learnerOwner = 0
runner Worker.Shared _ _ _ = pure (Left (Worker.ProtocolFailure "Shared learning requires a joint inference and learning owner"))

borrowed :: Resident.Resident owner -> Runner
borrowed owner = Runner (\paths call -> fmap Observation.Acknowledged <$> Resident.run owner paths call)
