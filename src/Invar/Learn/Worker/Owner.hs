module Invar.Learn.Worker.Owner (Runner, withRunner, borrowed, run) where

import Data.ByteString.Char8 qualified as Bytes
import Invar.Learn.Worker qualified as Worker
import Invar.Learn.Worker.Observation qualified as Observation
import Invar.Learn.Worker.Resident qualified as Resident

newtype Runner = Runner {run :: forall kind (scope :: kind). Resident.Paths -> Worker.Call scope -> IO (Either Worker.Failure (Observation.Observation scope))}

withRunner :: Worker.Mode -> Worker.Worker -> (Runner -> IO value) -> IO (Either Worker.Failure value)
withRunner Worker.Process worker action = Right <$> action (Runner execute)
  where
    execute paths call = fmap Observation.Terminated <$> Worker.run worker {Worker.checkpoint = Resident.checkpoint paths, Worker.output = Resident.output paths} call
withRunner Worker.Resident worker action =
    Resident.withResident (Resident.Options worker learnerOwner Bytes.putStrLn) $ \owner ->
        Right <$> action (Runner (\paths call -> fmap Observation.Acknowledged <$> Resident.run owner paths call))
  where
    learnerOwner = 0
withRunner Worker.Shared _ _ = pure (Left (Worker.ProtocolFailure "Shared learning requires a joint inference and learning owner"))

borrowed :: Resident.Resident owner -> Runner
borrowed owner = Runner (\paths call -> fmap Observation.Acknowledged <$> Resident.run owner paths call)
