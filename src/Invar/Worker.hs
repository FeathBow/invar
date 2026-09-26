module Invar.Worker (Worker (..), Batch.Reference (..), Failure (..), Execution, report, completion, loaded, run, runBatch, runSession, runBatchedSession) where

import Control.Exception (bracket, mask_)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Functor (void)
import Data.IORef (IORef, modifyIORef', newIORef, readIORef, writeIORef)
import Data.Text qualified as Text
import Data.Text.Encoding (encodeUtf8)
import Invar.Infer.Batch qualified as Batch
import Invar.Infer.Invocation qualified as I
import Invar.Infer.Result qualified as R
import Invar.Process qualified as Process
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as L
import System.Exit (ExitCode)
import System.IO (hFlush, stdout)

data Worker = Worker
    { executable :: FilePath
    , script :: FilePath
    , cache :: FilePath
    , adapter :: FilePath
    , environment :: [(String, String)]
    , configuration :: Maybe FilePath
    }

data Failure = WorkerExit ExitCode | InvalidOutput I.Error | ProtocolFailure String
    deriving (Eq, Show)

data Execution = Execution V.Completion R.Result L.Fact

data Pending = Pending I.Call (IORef (Maybe I.Permit))

report :: Execution -> R.Result
report (Execution _ result _) = result

completion :: Execution -> V.Completion
completion (Execution completed _ _) = completed

loaded :: Execution -> L.Fact
loaded (Execution _ _ fact) = fact

run :: Worker -> I.Call -> IO (Either Failure Execution)
run worker call = withRegistry worker $ \registry -> do
    pending <- Pending call <$> newIORef Nothing
    let inputs = arguments worker ++ I.arguments call
        command = Process.Command (executable worker) inputs (environment worker) (encodeUtf8 (Text.pack (I.input call)))
    returned <- Process.run command (authorize registry pending)
    case first failure returned of
        Right output -> first InvalidOutput <$> observe pending output
        Left problem -> pure (Left problem)

runBatch :: Worker -> [I.Call] -> IO (Either Failure [Execution])
runBatch worker = runSession worker Nothing (\line -> Bytes.hPutStrLn stdout line >> hFlush stdout)

runSession :: Worker -> Maybe Batch.Reference -> (ByteString -> IO ()) -> [I.Call] -> IO (Either Failure [Execution])
runSession worker reference echo calls = withRegistry worker $ \registry -> do
    pending <- traverse (\call -> Pending call <$> newIORef Nothing) calls
    let inputs = arguments worker ++ foldMap (\declared -> ["--reference=" ++ Batch.location declared, "--reference-digest=" ++ Batch.identity declared]) reference
        launch = Process.Launch (executable worker) inputs (environment worker) echo
        exchange value@(Pending call _) = Process.Exchange (I.batchInput call) (authorize registry value) (fmap void . observe value)
    returned <- Process.batch launch (map exchange pending)
    case first failure returned of
        Right outputs -> fmap (first InvalidOutput . sequence) (traverse (uncurry observe) (zip pending outputs))
        Left problem -> pure (Left problem)

runBatchedSession :: Worker -> Maybe Batch.Reference -> (ByteString -> IO ()) -> [I.Call] -> IO (Either Failure [Execution])
runBatchedSession _ _ _ [] = pure (Right [])
runBatchedSession worker reference echo calls = withRegistry worker $ \registry -> do
    slot <- newIORef Nothing
    let launch = Process.Launch (executable worker) (batchArguments worker) (environment worker) echo
        review output = do
            accepted <- grant registry slot (\current -> Batch.authorize current calls output)
            pure (Batch.permission <$> accepted)
        finish output = do
            permit <- readIORef slot
            pure $ case permit of
                Nothing -> Left (I.Protocol "Batch result has no accepted consumption permits")
                Just accepted -> map (\(completed, result, fact) -> Execution completed result fact) <$> Batch.observe accepted output
        exchange = Process.Exchange (Batch.input (adapter worker) reference calls) review (fmap void . finish)
    returned <- Process.batch launch [exchange]
    case first failure returned of
        Right [output] -> first InvalidOutput <$> finish output
        Right _ -> pure (Left (ProtocolFailure "Expected one finite batch response"))
        Left problem -> pure (Left problem)

arguments :: Worker -> [String]
arguments worker = [script worker, "--cache=" ++ cache worker, "--adapter=" ++ adapter worker] ++ maybe [] (\path -> ["--config=" ++ path]) (configuration worker)

batchArguments :: Worker -> [String]
batchArguments worker = [script worker, "--cache=" ++ cache worker] ++ maybe [] (\path -> ["--config=" ++ path]) (configuration worker)

withRegistry :: Worker -> (IORef L.Registry -> IO value) -> IO value
withRegistry _ = bracket (newIORef L.empty) (`modifyIORef'` L.close)

authorize :: IORef L.Registry -> Pending -> ByteString -> IO (Either I.Error ByteString)
authorize owner (Pending call slot) output = do
    accepted <- grant owner slot (\registry -> I.authorize registry call output)
    pure (I.permission <$> accepted)

grant :: IORef L.Registry -> IORef (Maybe permit) -> (L.Registry -> Either I.Error (L.Registry, permit)) -> IO (Either I.Error permit)
grant owner slot admit = mask_ $ do
    previous <- readIORef slot
    registry <- readIORef owner
    case previous of
        Just _ -> pure (Left (I.Protocol "Invocation already holds a consumption permit"))
        Nothing -> case admit registry of
            Left problem -> pure (Left problem)
            Right (updated, permit) -> do
                writeIORef owner updated
                writeIORef slot (Just permit)
                pure (Right permit)

observe :: Pending -> ByteString -> IO (Either I.Error Execution)
observe (Pending _ slot) output = do
    accepted <- readIORef slot
    pure $ case accepted of
        Nothing -> Left (I.Protocol "Inference result has no accepted load and consumption permit")
        Just permit -> completed permit <$> I.observe permit output
  where
    completed permit (result, value) = Execution result value (I.loadFact permit)

failure :: Process.Failure I.Error -> Failure
failure (Process.Exit status) = WorkerExit status
failure (Process.Rejected problem) = InvalidOutput problem
failure (Process.Protocol problem) = ProtocolFailure problem
