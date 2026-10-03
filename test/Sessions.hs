{-# LANGUAGE OverloadedStrings #-}

module Sessions (sessions) where

import BatchCalls (exchange, prepared, quote)
import Calls qualified as Fixture
import Control.Concurrent (forkIO, threadDelay)
import Control.Concurrent.MVar (newEmptyMVar, putMVar, takeMVar)
import Control.Monad (forM_, void)
import Data.Aeson (Value)
import Data.IORef (atomicModifyIORef', newIORef, readIORef)
import Data.List (sort, sortOn)
import Data.Maybe (isNothing)
import Hedgehog
import Invar.Cohort qualified as C
import Invar.Infer qualified as I
import Invar.Infer.Invocation qualified as V
import Invar.Infer.Result qualified as Result
import Invar.Reward qualified as Reward
import Invar.Rollout qualified as R
import Invar.Spec.Invocation qualified as B
import Invar.Transcript qualified as Transcript
import Invar.Worker qualified as W
import Numeric.Natural (Natural)
import Store (workspace)
import System.Exit (ExitCode (..))
import System.FilePath ((</>))
import System.Timeout (timeout)

sessions :: Group
sessions =
    Group
        "Session partition of a cohort"
        [ ("several sessions produce the batch of one session", once identical)
        , ("a failing or missing session fails the cohort", once failing)
        , ("cancelling the cohort stops every session", once cancelled)
        , ("cancelling the cohort after one session returned stops the others", once returned)
        , ("each request is reported when dispatched to a session and each result after its check", once observed)
        , ("a session without requests opens no transcript, and every opened transcript learns its outcome", once transcribed)
        ]
  where
    once = withTests 1 . property

members :: Natural
members = 5

execution :: [Natural]
execution = [4, 0, 3, 1, 2]

arrival :: [Natural]
arrival = [2, 4, 0, 1, 3]

assignment :: Int -> [[Natural]]
assignment count = [[execution !! position | position <- [0 .. fromIntegral members - 1], position `mod` count == slot] | slot <- [0 .. count - 1]]

setup :: Int -> PropertyT IO ([(V.Call, [Value])], C.Definition)
setup count = do
    (_, events) <- Fixture.setup
    planned <- evalEither (I.prepare Fixture.request)
    expected <- evalEither (Reward.decimal "#### 12")
    calls <- traverse (uncurry (prepared planned events)) (concatMap chain (assignment count))
    let tasks = [C.Task ("member" ++ show index) "group" planned expected | index <- [0 .. members - 1]]
    pure (calls, C.Definition (replicate 64 'a') tasks)
  where
    chain chosen = zip chosen (Nothing : map Just chosen)

script :: FilePath -> [(V.Call, [Value])] -> Int -> String
script root calls count =
    unlines (opening ++ branches ++ ["*) exit 31;;", "esac"])
  where
    opening = ["printf '%s\\n' launched >> " ++ quote (root </> "launched"), "case \"${INVAR_TEST_SESSION:-single}\" in"]
    branches = [tag slot ++ ") " ++ concatMap (exchange root) chosen ++ "exit 0;;" | (slot, chosen) <- zip [0 :: Int ..] (grouped calls (assignment count))]
    tag slot = if count == 1 then "single" else show slot
    grouped remaining (chosen : rest) = let (here, later) = splitAt (length chosen) remaining in here : grouped later rest
    grouped _ [] = []

options :: FilePath -> Int -> PropertyT IO R.Options
options root count = do
    (calls, definition) <- setup count
    let path = root </> ("session" ++ show count ++ ".sh")
    evalIO (writeFile path (script root calls count))
    let worker = W.Worker "/bin/sh" path root "unused" [] Nothing
        overlays = if count == 1 then [[]] else [[("INVAR_TEST_SESSION", show slot)] | slot <- [0 .. count - 1]]
    pure R.Options {R.worker = worker, R.mode = R.Serial, R.sessions = overlays, R.definition = definition, R.order = execution, R.delivery = arrival, R.reference = Nothing}

identical :: PropertyT IO ()
identical = do
    root <- workspace
    single <- options root 1
    split <- options root 3
    first <- evalIO (R.withDriver (\driver -> fmap project <$> R.run driver single)) >>= evalEither
    second <- evalIO (R.withDriver (\driver -> fmap project <$> R.run driver split)) >>= evalEither
    first === second
    launched <- evalIO (readFile (root </> "launched"))
    length (lines launched) === 4
  where
    project batch = (map R.name (R.samples batch), map R.reward (R.samples batch), map (Result.behaviorBits . R.observation) (R.samples batch), map bindingOf (R.delivered batch))
    bindingOf (B.Binding (B.CallId call) (B.AttemptId attempt) (B.Instance instanceId)) = (call, attempt, instanceId)

failing :: PropertyT IO ()
failing = do
    root <- workspace
    split <- options root 2
    let broken = split {R.sessions = [[("INVAR_TEST_SESSION", "0")], [("INVAR_TEST_SESSION", "9")]]}
    outcome <- evalIO (R.withDriver (\driver -> void <$> R.run driver broken))
    case outcome of
        Left (R.Execution (W.WorkerExit (ExitFailure 31))) -> pure ()
        Left problem -> annotateShow problem >> failure
        Right _ -> failure
    result <- evalIO (R.withDriver (\driver -> void <$> R.run driver split {R.sessions = []}))
    case result of
        Left (R.Dispatch _) -> pure ()
        Left problem -> annotateShow problem >> failure
        Right _ -> failure

cancelled :: PropertyT IO ()
cancelled = do
    root <- workspace
    split <- options root 2
    let path = root </> "slow.sh"
        record = " >> " ++ quote (root </> "launched")
    evalIO (writeFile path (unlines ["printf '%s\\n' \"started $INVAR_TEST_SESSION\"" ++ record, "sleep 3", "printf '%s\\n' \"late $INVAR_TEST_SESSION\"" ++ record, "exit 0"]))
    let slow = split {R.worker = W.Worker "/bin/sh" path root "unused" [] Nothing}
    outcome <- evalIO (timeout 1000000 (R.withDriver (\driver -> void <$> R.run driver slow)))
    outcome === Nothing
    evalIO (threadDelay 3500000)
    launched <- evalIO (readFile (root </> "launched"))
    sort (lines launched) === ["started 0", "started 1"]

returned :: PropertyT IO ()
returned = do
    root <- workspace
    split <- options root 2
    let path = root </> "uneven.sh"
        record = " >> " ++ quote (root </> "launched")
    evalIO (writeFile path (unlines ["printf '%s\\n' \"started $INVAR_TEST_SESSION\"" ++ record, "test \"$INVAR_TEST_SESSION\" = 0 && exit 0", "sleep 3", "printf '%s\\n' \"late $INVAR_TEST_SESSION\"" ++ record, "exit 0"]))
    let uneven = split {R.worker = W.Worker "/bin/sh" path root "unused" [] Nothing}
    finished <- evalIO newEmptyMVar
    _ <- evalIO (forkIO (timeout 1000000 (R.withDriver (\driver -> void <$> R.run driver uneven)) >>= putMVar finished . isNothing))
    outcome <- evalIO (timeout 10000000 (takeMVar finished))
    outcome === Just True
    evalIO (threadDelay 3000000)
    launched <- evalIO (readFile (root </> "launched"))
    sort (lines launched) === ["started 0", "started 1"]

data Report = Dispatched Natural Natural B.Binding | Checked Natural B.Binding [Natural]
    deriving (Eq, Show)

observed :: PropertyT IO ()
observed = do
    forM_ [1, 3] $ \count -> do
        root <- workspace
        chosen <- options root count
        log' <- evalIO (newIORef [])
        let push report = atomicModifyIORef' log' (\reports -> (reports ++ [report], ()))
            observer = R.Observer (\slot requests -> mapM_ (\(index, binding) -> push (Dispatched slot index binding)) requests) (\index binding result -> push (Checked index binding (map fromIntegral (Result.behaviorBits result))))
        batch <- evalIO (R.withDriver (\driver -> fmap project <$> R.runObserved driver observer chosen)) >>= evalEither
        reports <- evalIO (readIORef log')
        let dispatched = [(index, (slot, binding)) | Dispatched slot index binding <- reports]
            checked = [(index, (binding, words32)) | Checked index binding words32 <- reports]
        sort (map fst dispatched) === [0 .. members - 1]
        sort (map fst checked) === [0 .. members - 1]
        [sort [index | (index, (slot, _)) <- dispatched, slot == fromIntegral position] | position <- [0 .. count - 1]] === map sort (assignment count)
        forM_ checked $ \(index, (binding, words32)) -> do
            (slot, sent) <- evalMaybe (lookup index dispatched)
            sent === binding
            let position report = length (takeWhile (/= report) reports)
            assert (position (Dispatched slot index sent) < position (Checked index binding words32))
        [words32 | index <- [0 .. members - 1], Just (_, words32) <- [lookup index checked]] === batch
  where
    project result = [map fromIntegral (Result.behaviorBits (R.observation sample)) | sample <- R.samples result]

transcribed :: PropertyT IO ()
transcribed = do
    root <- workspace
    chosen <- options root 7
    opened <- evalIO (newIORef [])
    ended <- evalIO (newIORef [])
    let open slot = do
            atomicModifyIORef' opened (\slots -> (slot : slots, ()))
            pure (Transcript.Transcript (const (pure ())) (const (pure ())) (\outcome -> atomicModifyIORef' ended (\outcomes -> ((slot, outcome) : outcomes, ()))))
    outcome <- evalIO (R.withRecordedDriver R.Serial (R.worker chosen, R.sessions chosen) open (\driver -> void <$> R.run driver chosen))
    evalEither outcome >>= evalEither
    slots <- evalIO (readIORef opened)
    sort slots === [0 .. members - 1]
    outcomes <- evalIO (readIORef ended)
    sortOn fst outcomes === [(slot, Transcript.Exited ExitSuccess Transcript.Complete) | slot <- [0 .. members - 1]]
    launched <- evalIO (readFile (root </> "launched"))
    length (lines launched) === fromIntegral members
