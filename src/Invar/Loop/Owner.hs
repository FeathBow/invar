module Invar.Loop.Owner (validate, withShared) where

import Control.Monad (unless, void)
import Data.Bifunctor (first)
import Data.ByteString.Char8 qualified as Bytes
import Invar.Learn.Worker qualified as Learn
import Invar.Learn.Worker.Owner qualified as Learner
import Invar.Learn.Worker.Resident qualified as Update
import Invar.Process qualified as Process
import Invar.Process.Resident qualified as Transport
import Invar.Resident qualified as Boundary
import Invar.Rollout.Resident qualified as Rollout
import Invar.Worker qualified as Infer
import Invar.Worker.Resident qualified as Inference

type Configuration = (Infer.Worker, Learn.Worker, [[(String, String)]])

validate :: Configuration -> Either Learn.Failure ()
validate (inference, learning, overlays) = first Learn.ProtocolFailure $ do
    unless (Infer.executable inference == Learn.executable learning && Infer.script inference == Learn.script learning && Infer.cache inference == Learn.cache learning) (Left "Shared roles require the same executable, script and model cache")
    unless (length overlays == 1) (Left "Shared execution requires exactly one physical session")

withShared :: Configuration -> (Rollout.Pool -> Learner.Runner -> IO value) -> IO (Either Learn.Failure value)
withShared configuration@(inference, learning, overlays) action = case validate configuration of
    Left problem -> pure (Left problem)
    Right () -> first failure <$> Transport.withResident launch closing execute
  where
    owner = Boundary.Owner Boundary.Shared 0
    arguments = [Infer.script inference, "--cache=" ++ Infer.cache inference, "--reference=" ++ Learn.reference learning, "--session=0", "--shared"] ++ maybe [] (\path -> ["--config=" ++ path]) (Infer.configuration inference)
    launch = Process.Launch (Infer.executable inference) arguments (concat overlays) Bytes.putStrLn
    closing process = Transport.Handshake (Boundary.close owner) $ \encoded -> do
        count <- Transport.groups process
        pure (first Learn.ProtocolFailure (void (Boundary.closed owner count encoded)))
    execute process = Inference.withBorrowed process owner $ \collector ->
        Update.withBorrowed process owner $ \updater ->
            Right <$> action (Rollout.borrowed (inference, overlays) collector) (Learner.borrowed updater)
    failure (Process.Exit status) = Learn.WorkerExit status
    failure (Process.Rejected problem) = problem
    failure (Process.Protocol problem) = Learn.ProtocolFailure problem
