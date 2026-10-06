{-# LANGUAGE RoleAnnotations #-}

module Invar.Rollout (Driver, Mode (..), Options (..), Observer (..), Batch, Sample, Error (..), withDriver, withConfiguredDriver, withRecordedDriver, run, runObserved, silent, samples, delivered, name, group, trajectory, reward, scored) where

import Control.Concurrent (forkIOWithUnmask, killThread)
import Control.Concurrent.MVar (newEmptyMVar, newMVar, putMVar, readMVar, withMVar)
import Control.Exception (SomeException, finally, mask, onException, throwIO, try, uninterruptibleMask_)
import Data.Bifunctor (first)
import Data.IORef (modifyIORef', newIORef, readIORef)
import Data.List (sortOn)
import Invar.Cohort qualified as C
import Invar.Infer.Batch qualified as Batch
import Invar.Infer.Invocation qualified as I
import Invar.Infer.Trajectory (Trajectory)
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Reward qualified as Reward
import Invar.Rollout.Internal (Driver (..), reserve)
import Invar.Rollout.Resident qualified as Resident
import Invar.Schedule qualified as S
import Invar.Spec.Invocation qualified as V
import Invar.Transcript qualified as Transcript
import Invar.Worker qualified as W
import Numeric.Natural (Natural)

data Mode = Serial | Batched | Resident | Shared deriving (Eq, Show)

data Options = Options {worker :: W.Worker, mode :: Mode, sessions :: [[(String, String)]], definition :: C.Definition, order :: [Natural], delivery :: [Natural], reference :: Maybe Batch.Reference}

data Observer = Observer {dispatched :: Natural -> [(Natural, V.Binding)] -> IO (), checked :: Natural -> Trajectory -> IO ()}

silent :: Observer
silent = Observer (\_ _ -> pure ()) (\_ _ -> pure ())

type role Batch nominal
data Batch scope = Batch [Sample] [V.Binding]

data Sample = Sample C.Task Trajectory Reward.Scored

data Error = Declaration C.Error | Scheduling S.Error | Dispatch String | Preparation I.Error | Execution W.Failure | Admission C.Error
    deriving (Eq, Show)

withDriver :: (forall scope. Driver scope -> IO result) -> IO result
withDriver = driven Nothing Nothing

driven :: Maybe Resident.Pool -> Maybe (Natural -> IO Transcript.Transcript) -> (forall scope. Driver scope -> IO result) -> IO result
driven pool recorded action = do
    lock <- newMVar ()
    counter <- newIORef 0
    action (Driver lock counter pool recorded)

withConfiguredDriver :: Mode -> (W.Worker, [[(String, String)]]) -> (forall scope. Driver scope -> IO value) -> IO (Either Error value)
withConfiguredDriver mode configuration = configured mode configuration Nothing

withRecordedDriver :: Mode -> (W.Worker, [[(String, String)]]) -> (Natural -> IO Transcript.Transcript) -> (forall scope. Driver scope -> IO value) -> IO (Either Error value)
withRecordedDriver mode configuration opened = configured mode configuration (Just opened)

configured :: Mode -> (W.Worker, [[(String, String)]]) -> Maybe (Natural -> IO Transcript.Transcript) -> (forall scope. Driver scope -> IO value) -> IO (Either Error value)
configured Resident configuration@(_, overlays) recorded action
    | null overlays = pure (Left (Dispatch "At least one session is required"))
    | otherwise = first Execution <$> Resident.withPool configuration recorded (\pool -> driven (Just pool) recorded action)
configured Shared _ _ _ = pure (Left (Dispatch "Shared inference requires a joint inference and learning owner"))
configured _ _ recorded action = Right <$> driven Nothing recorded action

run :: Driver scope -> Options -> IO (Either Error (Batch scope))
run driver = runObserved driver silent

runObserved :: Driver scope -> Observer -> Options -> IO (Either Error (Batch scope))
runObserved driver observer options = case C.withCohort (definition options) (collect driver observer options) of
    Left problem -> pure (Left (Declaration problem))
    Right action -> action

collect :: Driver scope -> Observer -> Options -> C.Cohort cohort -> IO (Either Error (Batch scope))
collect driver@(Driver lock _ _ _) observer options cohort = case planning of
    Left problem -> pure (Left problem)
    Right (plan, selected) -> withMVar lock $ \() -> do
        let count = fromIntegral (length selected)
        base <- reserve driver count
        completed <- execute driver observer options (base, selected)
        pure $ do
            values <- completed
            supplied <- first Scheduling (S.deliver plan [(index, (observed, executed)) | (index, observed, executed) <- values])
            finish (definition options) cohort [(index, observed, executed) | (index, (observed, executed)) <- supplied]
  where
    planning = do
        let members = C.members cohort
        plan <- first Scheduling (S.prepare (fromIntegral (length members)) (order options) (delivery options))
        selected <- first Scheduling (S.execute plan members)
        pure (plan, selected)

execute :: Driver driver -> Observer -> Options -> (Natural, [(Natural, C.Member scope)]) -> IO (Either Error [(Natural, C.Observation scope, Trajectory)])
execute driver@(Driver _ _ pool recorded) observer options (base, selected) = case prepared of
    Left problem -> pure (Left problem)
    Right (calls, workers) -> dispatch calls workers `finally` mapM_ Resident.flush pool
  where
    prepared = (,) <$> traverse (prepareCall base) selected <*> runners driver options
    dispatch _ [] = pure (Left (Dispatch "At least one session is required"))
    dispatch calls [single] = do
        returned <- started single 0 (pure Transcript.standard) (zip selected calls)
        report (finishSession selected returned)
    dispatch calls workers = do
        let assigned = partition (length workers) (zip selected calls)
        outcomes <- concurrently [session slot launch members | (slot, launch, members) <- zip3 [0 ..] workers assigned]
        mapM_ (mapM_ Transcript.live . fst) outcomes
        pure $ do
            completed <- traverse snd outcomes
            pure (sortOn (\(index, _, _) -> index) (concat completed))
    session slot launch members = do
        buffer <- newIORef []
        returned <- started launch slot (pure (Transcript.echoing (\line -> modifyIORef' buffer (line :)))) members
        emitted <- reverse <$> readIORef buffer
        finished <- report (finishSession (map fst members) returned)
        pure (emitted, finished)
    started (Finite _) _ _ [] = pure (Right [])
    started (Finite launch) slot standard members = do
        transcript <- maybe standard ($ slot) recorded
        dispatched observer slot [(index, I.binding call) | ((index, _), call) <- members]
        launch transcript (map snd members)
    started (Owned launch) slot _ members = do
        dispatched observer slot [(index, I.binding call) | ((index, _), call) <- members]
        launch (map snd members)
    report finished = do
        mapM_ (mapM_ (\(index, _, executed) -> checked observer index executed)) finished
        pure finished

data Runner = Finite (Transcript.Transcript -> [I.Call] -> IO (Either W.Failure [Trajectory])) | Owned ([I.Call] -> IO (Either W.Failure [Trajectory]))

runners :: Driver scope -> Options -> Either Error [Runner]
runners (Driver _ _ pool _) options = case (mode options, pool) of
    (selected, Just owned)
        | selected `elem` [Resident, Shared] ->
            if Resident.matches owned (worker options, sessions options)
                then Right [Owned (launch (W.adapter (worker options)) (reference options)) | launch <- Resident.sessions owned]
                else Left (Dispatch "Resident launch configuration differs from its owning driver")
    (Resident, Nothing) -> Left (Dispatch "Resident execution requires a configured owning driver")
    (Shared, Nothing) -> Left (Dispatch "Shared execution requires a joint inference and learning owner")
    (_, Just _) -> Left (Dispatch "Resident owning driver cannot switch execution mode")
    (Serial, Nothing) -> Right [finite (W.runSession, overlay) | overlay <- sessions options]
    (Batched, Nothing) -> Right [finite (W.runBatchedSession, overlay) | overlay <- sessions options]
  where
    finite (launch, overlay) = Finite (launch ((worker options) {W.environment = overlay}) (reference options))

prepareCall :: Natural -> (Natural, C.Member scope) -> Either Error I.Call
prepareCall base (index, member) =
    let identity = base + index
        bound = V.ordinal identity
     in first Preparation (I.prepare bound (C.planned member))

finishSession :: [(Natural, C.Member scope)] -> Either W.Failure [Trajectory] -> Either Error [(Natural, C.Observation scope, Trajectory)]
finishSession selected returned = do
    completed <- first Execution returned
    if length completed == length selected
        then traverse record (zip selected completed)
        else Left (Execution (W.ProtocolFailure "Batch response count differs from selected requests"))
  where
    record ((index, member), completed) = do
        observed <- first Admission (C.record member completed)
        pure (index, observed, completed)

partition :: Int -> [value] -> [[value]]
partition count values = [[value | (position, value) <- zip [0 :: Int ..] values, position `mod` count == slot] | slot <- [0 .. count - 1]]

concurrently :: [IO value] -> IO [value]
concurrently actions = mask $ \restore -> do
    launched <- traverse launch actions
    outcomes <- restore (traverse (readMVar . snd) launched) `onException` recall launched
    traverse (either throwIO pure) outcomes
  where
    launch action = do
        box <- newEmptyMVar
        thread <- forkIOWithUnmask (\unmask -> attempt (unmask action) >>= putMVar box)
        pure (thread, box)
    attempt :: IO value -> IO (Either SomeException value)
    attempt = try
    recall launched = uninterruptibleMask_ (mapM_ (killThread . fst) launched >> mapM_ (readMVar . snd) launched)

finish :: C.Definition -> C.Cohort cohort -> [(Natural, C.Observation cohort, Trajectory)] -> Either Error (Batch scope)
finish definition cohort completed = do
    _ <- first Admission (C.admit cohort [observed | (_, observed, _) <- completed])
    let logical = zipWith sample (C.tasks definition) (sortOn (\(index, _, _) -> index) completed)
        arrival = [Trajectory.binding executed | (_, _, executed) <- completed]
    pure (Batch logical arrival)
  where
    sample task (_, observed, executed) = Sample task executed (C.scored observed)

samples :: Batch scope -> [Sample]
samples (Batch values _) = values

delivered :: Batch scope -> [V.Binding]
delivered (Batch _ bindings) = bindings

name :: Sample -> String
name (Sample task _ _) = C.name task

group :: Sample -> String
group (Sample task _ _) = C.group task

reward :: Sample -> Rational
reward = Reward.value . scored

scored :: Sample -> Reward.Scored
scored (Sample _ _ evaluated) = evaluated

trajectory :: Sample -> Trajectory
trajectory (Sample _ executed _) = executed
