{-# LANGUAGE OverloadedStrings #-}

module Rollouts (rollouts, options, reports) where

import Control.Monad (forM_, void)
import Data.Aeson (Object, eitherDecodeStrict, withObject, (.:))
import Data.Aeson.Types (parseEither)
import Data.ByteString.Char8 qualified as Bytes
import Hedgehog
import Invar.Cohort qualified as C
import Invar.Infer qualified as I
import Invar.Reward qualified as Reward
import Invar.Rollout qualified as R
import Invar.Schedule qualified as S
import Invar.Worker qualified as W
import Numeric.Natural (Natural)
import Store (workspace)
import System.Directory (doesPathExist, makeAbsolute)
import System.Exit (ExitCode (ExitFailure))
import System.FilePath ((</>))
import System.IO.Error (isDoesNotExistError, tryIOError)

rollouts :: Group
rollouts = Group "Actual rollout failure boundaries" [("invalid cohorts and schedules never launch a process", once validation), ("failed cohorts stop and never reuse reserved identities", once reserved), ("launch exceptions do not roll back identity reservation", once interrupted)]
  where
    once = withTests 1 . property

options :: FilePath -> PropertyT IO R.Options
options root = do
    let identity = replicate 64 'a'
    planned <- evalEither (I.prepare (I.Request identity (replicate 64 'c') (replicate 64 'e') (replicate 64 'f') "Same request, distinct members." 2 0.8 17))
    expected <- evalEither (Reward.decimal "#### 12")
    script <- evalIO (makeAbsolute "test/fail.sh")
    let worker = W.Worker "/bin/sh" script root "unused" [] Nothing
        tasks = [C.Task "first" "question" planned expected, C.Task "second" "question" planned expected]
    pure R.Options {R.worker = worker, R.mode = R.Serial, R.sessions = [[]], R.definition = C.Definition identity tasks, R.order = [0, 1], R.delivery = [0, 1], R.reference = Nothing}

run :: R.Driver scope -> R.Options -> IO (Either R.Error ())
run driver settings = void <$> R.run driver settings

validation :: PropertyT IO ()
validation = do
    root <- workspace
    settings <- options root
    let unavailable = settings {R.worker = (R.worker settings) {W.executable = root </> "missing"}}
        empty = unavailable {R.definition = (R.definition unavailable) {C.tasks = []}}
        orders = [[], [0], [0, 0], [0, 1, 2], [0, 2]]
    (declaration, scheduling) <- evalIO $ R.withDriver $ \driver -> do
        declared <- run driver empty
        scheduled <- traverse (\selected -> run driver unavailable {R.order = selected}) orders
        delivered <- traverse (\selected -> run driver unavailable {R.delivery = selected}) orders
        pure (declared, (scheduled, delivered))
    declaration === Left (R.Declaration C.EmptyCohort)
    forM_ (fst scheduling) (=== Left (R.Scheduling (S.InvalidPermutation S.Execution)))
    forM_ (snd scheduling) (=== Left (R.Scheduling (S.InvalidPermutation S.Delivery)))
    evalIO (doesPathExist (root </> "calls")) >>= (=== False)

reserved :: PropertyT IO ()
reserved = do
    root <- workspace
    settings <- options root
    outcomes <- evalIO $ R.withDriver $ \driver ->
        traverse (\selected -> run driver settings {R.order = selected}) [[1, 0], [0, 1], [1, 0]]
    forM_ outcomes (=== Left (R.Execution (W.WorkerExit (ExitFailure 7))))
    recorded <- reports root
    recorded === [(1, 1, 1), (2, 2, 2), (5, 5, 5)]

interrupted :: PropertyT IO ()
interrupted = do
    root <- workspace
    settings <- options root
    (exception, outcome) <- evalIO $ R.withDriver $ \driver -> do
        attempted <- tryIOError (run driver settings {R.worker = (R.worker settings) {W.executable = root </> "missing"}})
        next <- run driver settings
        pure (attempted, next)
    case exception of
        Left problem -> assert (isDoesNotExistError problem)
        Right _ -> failure
    outcome === Left (R.Execution (W.WorkerExit (ExitFailure 7)))
    reports root >>= (=== [(2, 2, 2)])

reports :: FilePath -> PropertyT IO [(Natural, Natural, Natural)]
reports root = do
    encoded <- evalIO (Bytes.readFile (root </> "calls"))
    traverse parse (Bytes.lines encoded)
  where
    parse encoded = do
        value <- evalEither (eitherDecodeStrict encoded :: Either String Object)
        evalEither (parseEither (\entry -> entry .: "binding" >>= withObject "binding" fields) value)
    fields value = (,,) <$> value .: "call" <*> value .: "attempt" <*> value .: "instance"
