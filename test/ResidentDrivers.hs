{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}

module ResidentDrivers (residentDrivers) where

import Calls qualified
import Control.Monad (forM_, void)
import Data.Aeson (Value (..))
import Data.IORef (newIORef, readIORef, writeIORef)
import Data.List (nub)
import Hedgehog
import Invar.Infer.Result qualified as Result
import Invar.Rollout qualified as Rollout
import Invar.Spec.Invocation qualified as Invocation
import Invar.Spec.Load qualified as Load
import Invar.Worker qualified as Worker
import Numeric.Natural (Natural)
import ResidentFixture qualified as Resident
import ResidentWorkloads qualified as F
import Store (workspace)
import System.Directory (doesFileExist)
import System.Exit (ExitCode (ExitFailure))
import System.FilePath ((</>))
import System.IO.Error (isDoesNotExistError, tryIOError)
import System.Posix.Signals (nullSignal, signalProcess)

residentDrivers :: Group
residentDrivers =
    Group
        "Resident owning rollout driver"
        [ ("one configured device pool spans cohorts policy changes and an unused owner", once sustained)
        , ("resident execution rejects absent owners and changed launch configuration", once configured)
        , ("a poisoned owner cannot dispatch again or complete its driver", once poisoned)
        , ("late close failure is returned after every other owner closes", once closing)
        , ("callback interruption terminates every owned child", once interrupted)
        , ("an escaped driver action cannot reopen a completed owner", once escaped)
        ]
  where
    once = withTests 1 . property

policies :: [String]
policies = map (replicate 64) ['a', 'd', 'a']

withDriver :: F.Fixture -> (forall scope. Rollout.Driver scope -> IO value) -> IO (Either Rollout.Error value)
withDriver fixture = Rollout.withConfiguredDriver Rollout.Resident (Rollout.worker first, Rollout.sessions first)
  where
    first = F.initial fixture

sustained :: PropertyT IO ()
sustained = forM_ [1, 2, 4] $ \count -> do
    root <- workspace
    fixture <- F.setup root count policies
    outcomes <- evalIO (withDriver fixture (\driver -> traverse (fmap (fmap project) . Rollout.run driver) (F.options fixture))) >>= evalEither
    completed <- traverse evalEither outcomes
    forM_ (zip [0 :: Natural ..] completed) $ \(index, (names, rewards, observed, delivered, loaded, truncated)) -> do
        names === ["member0", "member1", "member2"]
        truncated === replicate (fromIntegral F.memberCount) True
        rewards === replicate (fromIntegral F.memberCount) 0
        observed === replicate (fromIntegral F.memberCount) [0xbf000000, 0xbe800000]
        delivered === map (bound . (index * F.memberCount +)) F.arrival
        loaded === map (bound . (index * F.memberCount +)) [0 .. F.memberCount - 1]
    pids <- traverse (evalIO . readFile . (</> "pids") . fst) (F.owners fixture)
    assert (all ((== 1) . length . lines) pids)
    length (nub pids) === count
    forM_ (F.owners fixture) $ \(path, _) -> evalIO (doesFileExist (path </> "closed")) >>= (=== True)
  where
    project batch = (map Rollout.name values, map Rollout.reward values, map (Result.behaviorBits . Rollout.observation) values, Rollout.delivered batch, map (Invocation.completedBinding . Load.report . Rollout.loaded) values, map (Result.truncated . Rollout.observation) values)
      where
        values = Rollout.samples batch

bound :: Natural -> Invocation.Binding
bound index = Invocation.Binding (Invocation.CallId index) (Invocation.AttemptId index) (Invocation.Instance index)

configured :: PropertyT IO ()
configured = do
    root <- workspace
    original <- F.setup root 2 (take 1 policies)
    let fixture = original {F.owners = zipWith (\index (path, _) -> (path, Resident.scenarioWith index [])) [0 ..] (F.owners original)}
        first = F.initial fixture
        worker = Rollout.worker first
        workers = [worker {Worker.executable = "wrong"}, worker {Worker.script = "wrong"}, worker {Worker.cache = "wrong"}, worker {Worker.configuration = Nothing}]
        changed = first {Rollout.mode = Rollout.Serial} : first {Rollout.sessions = reverse (Rollout.sessions first)} : first {Rollout.sessions = []} : map (\selected -> first {Rollout.worker = selected}) workers
    F.writeScripts fixture
    evalIO (Rollout.withDriver (\driver -> void <$> Rollout.run driver first)) >>= dispatchFailure
    outcomes <- evalIO (withDriver fixture (\driver -> traverse (fmap void . Rollout.run driver) changed)) >>= evalEither
    forM_ outcomes dispatchFailure
    forM_ (F.owners fixture) $ \(path, _) -> do
        evalIO (doesFileExist (path </> "approved0")) >>= (=== False)
        evalIO (doesFileExist (path </> "closed")) >>= (=== True)

poisoned :: PropertyT IO ()
poisoned = do
    root <- workspace
    original <- F.setup root 1 (take 2 policies)
    let corrupt (path, scenario) = case Resident.groups scenario of
            first : rest -> (path, scenario {Resident.groups = first {Resident.released = Calls.change "owner" Null (Resident.released first)} : rest})
            [] -> error "Expected an active resident owner"
        fixture = original {F.owners = map corrupt (F.owners original)}
    F.writeScripts fixture
    captured <- evalIO (newIORef [])
    returned <- evalIO $ withDriver fixture $ \driver -> do
        outcomes <- traverse (fmap void . Rollout.run driver) (F.options fixture)
        writeIORef captured outcomes
    executionFailure returned
    evalIO (readIORef captured) >>= mapM_ executionFailure
    evalIO (length . lines <$> readFile (F.ownerRoot root 0 </> "pids")) >>= (=== 1)
    evalIO (doesFileExist (F.ownerRoot root 0 </> "approved0")) >>= (=== True)
    evalIO (doesFileExist (F.ownerRoot root 0 </> "approved1")) >>= (=== False)

closing :: PropertyT IO ()
closing = do
    root <- workspace
    original <- F.setup root 2 policies
    let alter index (path, scenario) = (path, if index == 1 then scenario {Resident.ending = "IFS= read -r extra && exit 29\nexit 7"} else scenario)
        fixture = original {F.owners = zipWith alter [0 :: Int ..] (F.owners original)}
    F.writeScripts fixture
    captured <- evalIO (newIORef [])
    returned <- evalIO $ withDriver fixture $ \driver -> do
        outcomes <- traverse (fmap void . Rollout.run driver) (F.options fixture)
        writeIORef captured outcomes
    returned === Left (Rollout.Execution (Worker.WorkerExit (ExitFailure 7)))
    evalIO (readIORef captured) >>= (=== replicate (length policies) (Right ()))
    forM_ (F.owners fixture) $ \(path, _) -> evalIO (doesFileExist (path </> "closed")) >>= (=== True)

interrupted :: PropertyT IO ()
interrupted = do
    root <- workspace
    fixture <- F.setup root 2 policies
    returned <- evalIO $ tryIOError $ withDriver fixture $ \driver -> do
        completed <- Rollout.run driver (F.initial fixture)
        _ <- either (ioError . userError . show) pure completed
        ioError (userError "Interrupt resident owner") :: IO ()
    case returned of
        Left _ -> success
        Right _ -> failure
    forM_ (F.owners fixture) $ \(path, _) -> do
        pid <- evalIO (read <$> readFile (path </> "pids"))
        stopped <- evalIO (tryIOError (signalProcess nullSignal pid))
        case stopped of
            Left problem -> assert (isDoesNotExistError problem)
            Right _ -> failure

escaped :: PropertyT IO ()
escaped = do
    root <- workspace
    original <- F.setup root 1 (take 1 policies)
    let fixture = original {F.owners = [(path, Resident.scenarioWith 0 []) | (path, _) <- F.owners original]}
    F.writeScripts fixture
    action <- evalIO (withDriver fixture (\driver -> pure (void <$> Rollout.run driver (F.initial fixture)))) >>= evalEither
    evalIO action >>= executionFailure
    evalIO (length . lines <$> readFile (F.ownerRoot root 0 </> "pids")) >>= (=== 1)

dispatchFailure :: Either Rollout.Error value -> PropertyT IO ()
dispatchFailure (Left (Rollout.Dispatch _)) = success
dispatchFailure _ = failure

executionFailure :: Either Rollout.Error value -> PropertyT IO ()
executionFailure (Left (Rollout.Execution _)) = success
executionFailure _ = failure
