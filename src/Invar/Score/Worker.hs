module Invar.Score.Worker (Failure (..), run) where

import Data.IORef (newIORef, readIORef, writeIORef)
import Invar.Process qualified as Process
import Invar.Score qualified as Score
import Invar.Spec.Load qualified as Load
import Invar.Worker qualified as Worker
import System.Exit (ExitCode)

data Failure = WorkerExit ExitCode | InvalidOutput Score.Error | ProtocolFailure String
    deriving (Eq, Show)

run :: Worker.Worker -> Score.Call -> IO (Either Failure Score.Report)
run worker call = do
    slot <- newIORef Nothing
    let arguments =
            [Worker.script worker, "--cache=" ++ Worker.cache worker, "--adapter=" ++ Worker.adapter worker]
                ++ maybe [] (\path -> ["--config=" ++ path]) (Worker.configuration worker)
        command = Process.Command (Worker.executable worker) arguments (Worker.environment worker) (Score.input call)
        authorize bytes = do
            previous <- readIORef slot
            case previous of
                Just _ -> pure (Left (Score.Protocol "Score call already holds a consumption permit"))
                Nothing -> case Score.authorize Load.empty call bytes of
                    Left problem -> pure (Left problem)
                    Right (_, permitted) -> writeIORef slot (Just permitted) >> pure (Right (Score.permission permitted))
    returned <- Process.run command authorize
    case returned of
        Left (Process.Exit status) -> pure (Left (WorkerExit status))
        Left (Process.Rejected problem) -> pure (Left (InvalidOutput problem))
        Left (Process.Protocol problem) -> pure (Left (ProtocolFailure problem))
        Right output -> do
            permitted <- readIORef slot
            pure $ case permitted of
                Nothing -> Left (ProtocolFailure "Score process completed without an admitted consumption")
                Just permit -> either (Left . InvalidOutput) Right (Score.observe permit output)
