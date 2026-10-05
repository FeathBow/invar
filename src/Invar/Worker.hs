module Invar.Worker (Worker (..), Batch.Reference (..), Failure (..), run, runBatch, runSession, runBatchedSession, exchange, failure) where

import Control.Monad (foldM)
import Data.ByteString (ByteString)
import Data.Text qualified as Text
import Data.Text.Encoding (encodeUtf8)
import Invar.Infer.Batch qualified as Batch
import Invar.Infer.Invocation qualified as I
import Invar.Infer.Session qualified as Session
import Invar.Infer.Trajectory (Trajectory)
import Invar.Process qualified as Process
import Invar.Resident.Owner qualified as Owner
import Invar.Transcript qualified as Transcript
import System.Exit (ExitCode)

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

run :: Worker -> I.Call -> IO (Either Failure Trajectory)
run worker call = do
    let launch = Process.Launch (executable worker) (arguments worker ++ I.arguments call) (environment worker) Transcript.standard
    returned <- drive Session.Single launch (const (encodeUtf8 (Text.pack (I.input call)))) (Session.Declaration [call] Nothing)
    pure $ case returned of
        Right [single] -> Right single
        Right _ -> Left (ProtocolFailure "Expected one completed inference call")
        Left problem -> Left problem

runBatch :: Worker -> [I.Call] -> IO (Either Failure [Trajectory])
runBatch worker = runSession worker Nothing Transcript.standard

runSession :: Worker -> Maybe Batch.Reference -> Transcript.Transcript -> [I.Call] -> IO (Either Failure [Trajectory])
runSession _ _ _ [] = pure (Right [])
runSession worker reference transcript calls = drive Session.Serial launch (\(Session.Request selected) -> foldMap I.batchInput selected) (Session.Declaration calls (fmap Batch.identity reference))
  where
    inputs = arguments worker ++ foldMap (\declared -> ["--reference=" ++ Batch.location declared, "--reference-digest=" ++ Batch.identity declared]) reference
    launch = Process.Launch (executable worker) inputs (environment worker) transcript

runBatchedSession :: Worker -> Maybe Batch.Reference -> Transcript.Transcript -> [I.Call] -> IO (Either Failure [Trajectory])
runBatchedSession _ _ _ [] = pure (Right [])
runBatchedSession worker reference transcript calls = drive Session.Batched launch (\(Session.Request selected) -> Batch.input (adapter worker) reference selected) (Session.Declaration calls (fmap Batch.identity reference))
  where
    launch = Process.Launch (executable worker) (batchArguments worker) (environment worker) transcript

drive :: Session.Protocol -> Process.Launch -> (Session.Request -> ByteString) -> Session.Declaration -> IO (Either Failure [Trajectory])
drive protocol launch encode declaration = Process.withChannel launch $ \channel -> do
    returned <- exchange channel encode (Session.start protocol) (Session.Dispatched declaration)
    pure (either (Left . failure) (\(_, trajectories, _) -> Right trajectories) returned)

exchange :: Process.Channel -> (Session.Request -> ByteString) -> Session.Session -> Session.Input -> IO (Either Session.Error (Session.Session, [Trajectory], Maybe Owner.State))
exchange channel encode = step Nothing
  where
    step owned session supplied = case Session.step session supplied of
        Left problem -> case supplied of
            Session.Ended _ -> pure (Left problem)
            _ -> Process.halt channel >> pure (Left problem)
        Right (following, products) -> do
            (admitted, held) <- foldM perform (Nothing, owned) products
            case admitted of
                Just trajectories -> pure (Right (following, trajectories, held))
                Nothing -> do
                    received <- Process.receive channel
                    case received of
                        Process.Line value -> step held following (Session.Line value)
                        Process.Fragment value -> step held following (Session.Fragment value)
                        Process.Exhausted -> do
                            status <- Process.exited channel
                            step held following (Session.Ended (Transcript.Exited status Transcript.Complete))
    perform (found, held) produced = case produced of
        Session.SendRequest request -> Process.send channel (encode request) >> pure (found, held)
        Session.Send value -> Process.send channel value >> pure (found, held)
        Session.Close -> Process.shut channel >> pure (found, held)
        Session.Admitted trajectories -> pure (Just trajectories, held)
        Session.Owned current -> pure (found, Just current)

arguments :: Worker -> [String]
arguments worker = [script worker, "--cache=" ++ cache worker, "--adapter=" ++ adapter worker] ++ maybe [] (\path -> ["--config=" ++ path]) (configuration worker)

batchArguments :: Worker -> [String]
batchArguments worker = [script worker, "--cache=" ++ cache worker] ++ maybe [] (\path -> ["--config=" ++ path]) (configuration worker)

failure :: Session.Error -> Failure
failure (Session.Exited status) = WorkerExit status
failure (Session.Invalid problem) = InvalidOutput problem
failure (Session.Protocol problem) = ProtocolFailure problem
