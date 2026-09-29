{-# LANGUAGE OverloadedStrings #-}

module LearnerCalls (learnerCalls) where

import BatchCalls (quote)
import Control.Monad (forM_, void)
import Data.Aeson (Value (..), decodeStrict, object, toJSON, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseMaybe)
import Data.ByteString.Char8 qualified as Bytes
import Data.IORef (newIORef, readIORef, writeIORef)
import Data.Text qualified as Text
import Hedgehog
import Invar.Learn qualified as L
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Stream qualified as S
import Invar.Learn.Worker qualified as W
import Invar.Learn.Worker.Resident qualified as Resident
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as Load
import LearnerFixture qualified as F
import Policies qualified
import Store (workspace)
import System.Directory (doesFileExist)
import System.Exit (ExitCode (ExitFailure))
import System.FilePath ((</>))
import Updates (alter, change, field, wire)

learnerCalls :: Group
learnerCalls = Group "Resident learner admission" [("two acknowledged updates share a child and retain exact historical facts", once completed), ("all staged artifacts are checked before release", once artifacts), ("initial load and activation are mandatory before permission", once readiness), ("a step from another state, step or cotangent stops the update before release", once steps), ("a later update cannot replay a physical model load", once activation), ("a checked result requires one actual update measurement", once completion), ("mismatched release poisons the owner before another update", once acknowledgement), ("retired learner load instances cannot be reused", once replay), ("final closing failure propagates after acknowledged updates", once closing), ("escaped learner owners cannot start another process", once escaped), ("a process learner receives the core cotangents for every reported step", once processed)]
  where
    once = withTests 1 . property

binding :: Int -> V.Binding
binding index = let ordinal = fromIntegral index in V.Binding (V.CallId ordinal) (V.AttemptId ordinal) (V.Instance ordinal)

completed :: PropertyT IO ()
completed = do
    root <- workspace
    (returned, encoded, expected) <- F.withPlan root $ \planned -> do
        exchanges <- sequence [F.prepare root (0, binding 7) planned, F.prepare root (1, binding 8) planned]
        let selected = F.scenario exchanges
        (outcome, raw) <- F.run root selected (`F.execute` exchanges)
        pure (fmap (map (\receipt -> (P.completion (Resident.report receipt), Resident.loaded receipt, L.program (Resident.plan receipt), Resident.staged receipt, Resident.acknowledgement receipt))) outcome, raw, F.wire (concatMap (\exchange -> F.before exchange ++ F.steps exchange ++ F.after exchange ++ [F.released exchange]) exchanges ++ [F.closed selected]))
    receipts <- evalEither returned
    length receipts === 2
    forM_ (zip [0 :: Int ..] receipts) $ \(index, (result, fact, program, path, acknowledged)) -> do
        V.completedBinding result === binding (index + 7)
        V.completedBinding (Load.report fact) === binding (index + 7)
        V.completedProgram result === program
        assert (V.completedProgram (Load.report fact) /= program)
        path === root </> ("update" ++ show index)
        assert ("\"stage\":\"released\"" `Bytes.isInfixOf` acknowledged)
    encoded === expected
    evalIO (length . lines <$> readFile (root </> "learner-pids")) >>= (=== 1)
    evalIO (doesFileExist (root </> "learner-closed")) >>= assert

artifacts :: PropertyT IO ()
artifacts = forM_ ["gradients.safetensors", "probabilities.json", "adapter.safetensors", "learner.pt"] $ \name -> do
    root <- workspace
    returned <- F.withPlan root $ \planned -> do
        exchange <- F.prepare root (0, binding 7) planned
        let path = Resident.output (F.paths exchange) </> name
        if name == "adapter.safetensors"
            then Bytes.writeFile path (Policies.artifact [("fixture", [1], Bytes.pack ['\0', '\0', '\0', '\64'])])
            else Bytes.appendFile path "changed"
        (outcome, _) <- F.run root (F.scenario [exchange]) (\owner -> fmap void (F.execute owner [exchange]))
        pure outcome
    case returned of
        Left (W.GradientMismatch _ _) -> name === "gradients.safetensors"
        Left (W.ProbabilityMismatch _ _) -> name === "probabilities.json"
        Left (W.PolicyMismatch _ _) -> name === "adapter.safetensors"
        Left (W.LearnerMismatch _ _) -> name === "learner.pt"
        unexpected -> annotateShow unexpected >> failure
    evalIO (doesFileExist (root </> "learner-approved0")) >>= assert
    evalIO (doesFileExist (root </> "learner-released0")) >>= (=== False)

steps :: PropertyT IO ()
steps = forM_ changes $ \modify -> do
    root <- workspace
    returned <- F.withPlan root $ \planned -> do
        original <- F.prepare root (0, binding 7) planned
        let exchange = original {F.steps = modify (F.steps original)}
        fmap (void . fst) (F.run root (F.scenario [exchange]) (\owner -> F.execute owner [exchange]))
    case returned of
        Left (W.InvalidOutput _) -> success
        unexpected -> annotateShow unexpected >> failure
    evalIO (doesFileExist (root </> "learner-released0")) >>= (=== False)
  where
    changes = [alter 0 (change "binding" (object ["call" .= (9 :: Int), "attempt" .= (9 :: Int), "instance" .= (9 :: Int)])), alter 0 (change "state" (String (Text.replicate 64 "f"))), alter 0 (change "step" (Number 1)), \events -> take (length events - 1) events ++ [change "consumed" (toJSON ([] :: [Value])) (last events)], \events -> take (length events - 1) events ++ [change "after" (String (Text.replicate 64 "f")) (last events)], reverse]

readiness :: PropertyT IO ()
readiness = forM_ changes $ \modify -> do
    root <- workspace
    returned <- F.withPlan root $ \planned -> do
        original <- F.prepare root (0, binding 7) planned
        let exchange = original {F.before = modify (F.before original)}
        fmap (void . fst) (F.run root (F.scenario [exchange]) (\owner -> F.execute owner [exchange]))
    rejected returned
    evalIO (doesFileExist (root </> "learner-approved0")) >>= (=== False)
  where
    changes = [drop 1, \events -> take 1 events ++ drop 2 events, (F.timer "load" :), alter 1 (change "cpu_seconds" (Number (-1))), alter 2 (change "model" Null), \events -> take 2 events ++ drop 3 events, alter 3 (change "program" (String "another program"))]

activation :: PropertyT IO ()
activation = do
    root <- workspace
    returned <- F.withPlan root $ \planned -> do
        initial <- F.prepare root (0, binding 7) planned
        next <- F.prepare root (1, binding 8) planned
        let exchanges = [initial, next {F.before = F.timer "load" : F.before next}]
        fmap (void . fst) (F.run root (F.scenario exchanges) (`F.execute` exchanges))
    rejected returned
    evalIO (doesFileExist (root </> "learner-approved0")) >>= assert
    evalIO (doesFileExist (root </> "learner-approved1")) >>= (=== False)

completion :: PropertyT IO ()
completion = forM_ changes $ \modify -> do
    root <- workspace
    returned <- F.withPlan root $ \planned -> do
        original <- F.prepare root (0, binding 7) planned
        let exchange = original {F.after = modify (F.after original)}
        fmap (void . fst) (F.run root (F.scenario [exchange]) (\owner -> F.execute owner [exchange]))
    rejected returned
    evalIO (doesFileExist (root </> "learner-approved0")) >>= assert
    evalIO (doesFileExist (root </> "learner-released0")) >>= (=== False)
  where
    changes = [drop 1, (F.timer "reward_update" :), alter 0 (change "cpu_seconds" (Number (-1))), alter 1 (change "extra" Null), alter 1 (change "storage" (String "published"))]

acknowledgement :: PropertyT IO ()
acknowledgement = forM_ changes $ \modify -> do
    root <- workspace
    attempts <- evalIO (newIORef [])
    returned <- F.withPlan root $ \planned -> do
        original <- F.prepare root (0, binding 7) planned
        next <- F.prepare root (1, binding 8) planned
        let exchange = original {F.released = modify (F.released original)}
        fmap (void . fst) $ F.run root (F.scenario [exchange, next]) $ \owner -> do
            first <- void <$> Resident.run owner (F.paths exchange) (F.call exchange)
            second <- void <$> Resident.run owner (F.paths next) (F.call next)
            writeIORef attempts [first, second]
            pure (Right ())
    rejected returned
    actual <- evalIO (readIORef attempts)
    length actual === 2
    mapM_ rejected actual
    evalIO (doesFileExist (root </> "learner-released0")) >>= assert
    evalIO (doesFileExist (root </> "learner-approved1")) >>= (=== False)
    evalIO (length . lines <$> readFile (root </> "learner-pids")) >>= (=== 1)
  where
    changes = [change "owner" (object ["role" .= String "inference", "session" .= (0 :: Int)]), change "result_sha256" (String (Text.replicate 64 "f")), change "loads" (Array mempty), \value -> change "loads" (Array (pure (change "program" (String "different") (firstLoad value)))) value, change "stage" (String "closed"), change "measurement" (String ""), change "measurement" (String "{\"stage\":\"released\",\"cpu_seconds\":-1}\n")]
    firstLoad value = case field "loads" value of
        Array values -> case foldr (:) [] values of
            first : _ -> first
            [] -> error "Missing fixture load"
        _ -> error "Expected fixture loads"

replay :: PropertyT IO ()
replay = do
    root <- workspace
    returned <- F.withPlan root $ \planned -> do
        initial <- F.prepare root (0, binding 7) planned
        next <- F.prepare root (1, V.Binding (V.CallId 8) (V.AttemptId 8) (V.Instance 7)) planned
        let exchanges = [initial, next]
        fmap (void . fst) (F.run root (F.scenario exchanges) (`F.execute` exchanges))
    rejected returned
    evalIO (doesFileExist (root </> "learner-released0")) >>= assert
    evalIO (doesFileExist (root </> "learner-approved1")) >>= (=== False)

closing :: PropertyT IO ()
closing = forM_ [False, True] $ \badExit -> do
    root <- workspace
    completedCount <- evalIO (newIORef 0)
    returned <- F.withPlan root $ \planned -> do
        exchanges <- sequence [F.prepare root (0, binding 7) planned, F.prepare root (1, binding 8) planned]
        let initial = F.scenario exchanges
            selected = if badExit then initial {F.ending = "IFS= read -r extra && exit 29\nexit 7"} else initial {F.closed = change "groups" (Number 3) (F.closed initial)}
        fmap (void . fst) $ F.run root selected $ \owner -> do
            actual <- F.execute owner exchanges
            case actual of
                Left problem -> pure (Left problem)
                Right receipts -> writeIORef completedCount (length receipts) >> pure (Right ())
    rejected returned
    if badExit then returned === Left (W.WorkerExit (ExitFailure 7)) else success
    evalIO (readIORef completedCount) >>= (=== 2)
    evalIO (doesFileExist (root </> "learner-closed")) >>= assert

escaped :: PropertyT IO ()
escaped = do
    root <- workspace
    (closed, returned) <- F.withPlan root $ \planned -> do
        exchange <- F.prepare root (0, binding 7) planned
        saved <- newIORef (pure (Right ()))
        (outcome, _) <- F.run root (F.scenario []) $ \owner -> do
            writeIORef saved (void <$> Resident.run owner (F.paths exchange) (F.call exchange))
            pure (Right ())
        action <- readIORef saved
        actual <- action
        pure (outcome, actual)
    closed === Right ()
    returned === Left (W.ProtocolFailure "Resident worker is already closed")
    evalIO (length . lines <$> readFile (root </> "learner-pids")) >>= (=== 1)

rejected :: (Show value) => Either W.Failure value -> PropertyT IO ()
rejected (Left _) = success
rejected unexpected = annotateShow unexpected >> failure

processed :: PropertyT IO ()
processed = do
    root <- workspace
    (returned, replies, digests) <- F.withPlan root $ \planned -> do
        exchange <- F.prepare root (0, binding 7) planned
        let directory = Resident.output (F.paths exchange)
            worker = W.Worker "/bin/sh" (root </> "process.sh") root "process checkpoint" "process reference" directory
            kept = [event | event <- F.before exchange, stage event `elem` [Just "loaded_learner", Just "consumed"]]
            emit values = "printf '%s\\n' " ++ unwords (map (quote . Bytes.unpack) (Bytes.lines (wire values)))
            stepLines record = emit [record] : ["IFS= read -r reply || exit 23\nprintf '%s\\n' \"$reply\" >> " ++ quote (root </> "replies") | stage record == Just "current"]
            script = unlines (["IFS= read -r input || exit 21", emit kept, "IFS= read -r permission || exit 22"] ++ concatMap stepLines (F.steps exchange) ++ [emit (F.after exchange)])
            applied = [value | value <- F.steps exchange, stage value == Just "applied"]
        writeFile (W.script worker) script
        outcome <- W.run worker (F.call exchange)
        received <- Bytes.readFile (root </> "replies")
        pure (void outcome, received, concat [digests | value <- applied, Just digests <- [parseMaybe (withObject "applied" (.: "consumed")) value]])
    returned === Right ()
    let decoded = [reply | line <- Bytes.lines replies, Just reply <- [decodeStrict line]]
        reported = [S.digest (S.Reply 0 "" "" "" objective reward) | value <- decoded, Just (objective, reward) <- [parseMaybe (withObject "cotangents" (\fields -> (,) <$> fields .: "objective" <*> fields .: "reward")) value]]
    assert (not (null digests))
    reported === digests
  where
    stage (Object fields) = case Fields.lookup "stage" fields of
        Just (String name) -> Just name
        _ -> Nothing
    stage _ = Nothing
