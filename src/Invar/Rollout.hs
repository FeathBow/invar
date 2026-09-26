{-# LANGUAGE RoleAnnotations #-}

module Invar.Rollout (Driver, Mode (..), Options (..), Batch, Sample, Error (..), withDriver, withConfiguredDriver, run, samples, delivered, name, group, observation, reward, scored, completion, loaded) where

import Control.Concurrent (forkIOWithUnmask, killThread)
import Control.Concurrent.MVar (newEmptyMVar, newMVar, putMVar, takeMVar, withMVar)
import Control.Exception (SomeException, finally, mask, onException, throwIO, try, uninterruptibleMask_)
import Data.Bifunctor (first)
import Data.ByteString.Char8 qualified as Bytes
import Data.IORef (modifyIORef', newIORef, readIORef)
import Data.List (sortOn)
import Invar.Cohort qualified as C
import Invar.Infer.Batch qualified as Batch
import Invar.Infer.Invocation qualified as I
import Invar.Infer.Result qualified as R
import Invar.Reward qualified as Reward
import Invar.Rollout.Internal (Driver (..), reserve)
import Invar.Rollout.Observation qualified as Observed
import Invar.Rollout.Resident qualified as Resident
import Invar.Schedule qualified as S
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as Load
import Invar.Worker qualified as W
import Numeric.Natural (Natural)
import System.IO (hFlush, stdout)

data Mode = Serial | Batched | Resident | Shared deriving (Eq, Show)

data Options = Options {worker :: W.Worker, mode :: Mode, sessions :: [[(String, String)]], definition :: C.Definition, order :: [Natural], delivery :: [Natural], reference :: Maybe Batch.Reference}

type role Batch nominal
data Batch scope = Batch [Sample] [V.Binding]

data Sample = Sample C.Task Observed.Observation Reward.Scored

data Error = Declaration C.Error | Scheduling S.Error | Dispatch String | Preparation I.Error | Execution W.Failure | Admission C.Error
    deriving (Eq, Show)

withDriver :: (forall scope. Driver scope -> IO result) -> IO result
withDriver action = do
    lock <- newMVar ()
    counter <- newIORef 0
    action (Driver lock counter Nothing)

withConfiguredDriver :: Mode -> (W.Worker, [[(String, String)]]) -> (forall scope. Driver scope -> IO value) -> IO (Either Error value)
withConfiguredDriver Resident configuration@(_, overlays) action
    | null overlays = pure (Left (Dispatch "At least one session is required"))
    | otherwise = first Execution <$> Resident.withPool configuration echo (\pool -> withDriver (\(Driver lock counter _) -> action (Driver lock counter (Just pool))))
withConfiguredDriver Shared _ _ = pure (Left (Dispatch "Shared inference requires a joint inference and learning owner"))
withConfiguredDriver _ _ action = Right <$> withDriver action

run :: Driver scope -> Options -> IO (Either Error (Batch scope))
run driver options = case C.withCohort (definition options) (collect driver options) of
    Left problem -> pure (Left (Declaration problem))
    Right action -> action

collect :: Driver scope -> Options -> C.Cohort cohort -> IO (Either Error (Batch scope))
collect driver@(Driver lock _ _) options cohort = case planning of
    Left problem -> pure (Left problem)
    Right (plan, selected) -> withMVar lock $ \() -> do
        let count = fromIntegral (length selected)
        base <- reserve driver count
        completed <- execute driver options (base, selected)
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

execute :: Driver driver -> Options -> (Natural, [(Natural, C.Member scope)]) -> IO (Either Error [(Natural, C.Observation scope, Observed.Observation)])
execute driver@(Driver _ _ pool) options (base, selected) = case prepared of
    Left problem -> pure (Left problem)
    Right (calls, workers) -> dispatch calls workers `finally` mapM_ Resident.flush pool
  where
    prepared = (,) <$> traverse (prepareCall base) selected <*> runners driver options
    dispatch _ [] = pure (Left (Dispatch "At least one session is required"))
    dispatch calls [single] = finishSession selected <$> single echo calls
    dispatch calls workers = do
        let assigned = partition (length workers) (zip selected calls)
        outcomes <- concurrently [session launch members | (launch, members) <- zip workers assigned]
        mapM_ (mapM_ echo . fst) outcomes
        pure $ do
            completed <- traverse (\(_, (members, returned)) -> finishSession members returned) outcomes
            pure (sortOn (\(index, _, _) -> index) (concat completed))
    session launch members = do
        buffer <- newIORef []
        returned <- launch (\line -> modifyIORef' buffer (line :)) (map snd members)
        emitted <- reverse <$> readIORef buffer
        pure (emitted, (map fst members, returned))

type Runner = (Bytes.ByteString -> IO ()) -> [I.Call] -> IO (Either W.Failure [Observed.Observation])

runners :: Driver scope -> Options -> Either Error [Runner]
runners (Driver _ _ pool) options = case (mode options, pool) of
    (selected, Just owned)
        | selected `elem` [Resident, Shared] ->
            if Resident.matches owned (worker options, sessions options)
                then Right [\_ calls -> fmap (map Observed.Acknowledged) <$> launch (W.adapter (worker options)) (reference options) calls | launch <- Resident.sessions owned]
                else Left (Dispatch "Resident launch configuration differs from its owning driver")
    (Resident, Nothing) -> Left (Dispatch "Resident execution requires a configured owning driver")
    (Shared, Nothing) -> Left (Dispatch "Shared execution requires a joint inference and learning owner")
    (_, Just _) -> Left (Dispatch "Resident owning driver cannot switch execution mode")
    (Serial, Nothing) -> Right [finite (W.runSession, overlay) | overlay <- sessions options]
    (Batched, Nothing) -> Right [finite (W.runBatchedSession, overlay) | overlay <- sessions options]
  where
    finite (launch, overlay) emit calls = fmap (map Observed.Terminated) <$> launch ((worker options) {W.environment = overlay}) (reference options) emit calls

echo :: Bytes.ByteString -> IO ()
echo line = Bytes.hPutStrLn stdout line >> hFlush stdout

prepareCall :: Natural -> (Natural, C.Member scope) -> Either Error I.Call
prepareCall base (index, member) =
    let identity = base + index
        bound = V.ordinal identity
     in first Preparation (I.prepare bound (C.planned member))

finishSession :: [(Natural, C.Member scope)] -> Either W.Failure [Observed.Observation] -> Either Error [(Natural, C.Observation scope, Observed.Observation)]
finishSession selected returned = do
    completed <- first Execution returned
    if length completed == length selected
        then traverse record (zip selected completed)
        else Left (Execution (W.ProtocolFailure "Batch response count differs from selected requests"))
  where
    record ((index, member), completed) = do
        observed <- first Admission (C.record member (Observed.report completed))
        pure (index, observed, completed)

partition :: Int -> [value] -> [[value]]
partition count values = [[value | (position, value) <- zip [0 :: Int ..] values, position `mod` count == slot] | slot <- [0 .. count - 1]]

concurrently :: [IO value] -> IO [value]
concurrently actions = mask $ \restore -> do
    launched <- traverse launch actions
    outcomes <- restore (traverse (takeMVar . snd) launched) `onException` recall launched
    traverse (either throwIO pure) outcomes
  where
    launch action = do
        box <- newEmptyMVar
        thread <- forkIOWithUnmask (\unmask -> attempt (unmask action) >>= putMVar box)
        pure (thread, box)
    attempt :: IO value -> IO (Either SomeException value)
    attempt = try
    recall launched = uninterruptibleMask_ (mapM_ (killThread . fst) launched >> mapM_ (takeMVar . snd) launched)

finish :: C.Definition -> C.Cohort cohort -> [(Natural, C.Observation cohort, Observed.Observation)] -> Either Error (Batch scope)
finish definition cohort completed = do
    _ <- first Admission (C.admit cohort [observed | (_, observed, _) <- completed])
    let logical = zipWith sample (C.tasks definition) (sortOn (\(index, _, _) -> index) completed)
        arrival = [V.completedBinding (Observed.completion executed) | (_, _, executed) <- completed]
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

observation :: Sample -> R.Result
observation (Sample _ executed _) = Observed.report executed

reward :: Sample -> Rational
reward = Reward.value . scored

scored :: Sample -> Reward.Scored
scored (Sample _ _ evaluated) = evaluated

completion :: Sample -> V.Completion
completion (Sample _ executed _) = Observed.completion executed

loaded :: Sample -> Load.Fact
loaded (Sample _ executed _) = Observed.loaded executed
