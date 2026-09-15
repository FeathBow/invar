{-# LANGUAGE OverloadedStrings #-}

module Sessions (sessions) where

import BatchCalls (exchange, prepared, quote)
import Calls qualified as Fixture
import Control.Concurrent (threadDelay)
import Control.Monad (void)
import Data.Aeson (Value)
import Data.List (sort)
import Hedgehog
import Invar.Cohort qualified as C
import Invar.Infer qualified as I
import Invar.Infer.Invocation qualified as V
import Invar.Infer.Result qualified as Result
import Invar.Reward qualified as Reward
import Invar.Rollout qualified as R
import Invar.Spec.Invocation qualified as B
import Invar.Worker qualified as W
import Numeric.Natural (Natural)
import Store (workspace)
import System.Exit (ExitCode (ExitFailure))
import System.FilePath ((</>))
import System.Timeout (timeout)

sessions :: Group
sessions =
    Group
        "Session partition of a cohort"
        [ ("several sessions produce the batch of one session", once identical)
        , ("a failing or missing session fails the cohort", once failing)
        , ("cancelling the cohort stops every session", once cancelled)
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
    pure R.Options {R.worker = worker, R.mode = R.Serial, R.sessions = overlays, R.definition = definition, R.order = execution, R.delivery = arrival}

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
