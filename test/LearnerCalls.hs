{-# LANGUAGE OverloadedStrings #-}

module LearnerCalls (learnerCalls) where

import Control.Monad (forM_, void)
import Data.Aeson (Value (..), decodeStrict, object, toJSON, withObject, (.:), (.=))
import Data.Aeson.Types (parseMaybe)
import Data.ByteString.Char8 qualified as Bytes
import Data.IORef (modifyIORef', newIORef, readIORef, writeIORef)
import Data.Text qualified as Text
import Hedgehog
import Invar.Async.Completion qualified as Completion
import Invar.Learn qualified as L
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Stream qualified as S
import Invar.Learn.Worker qualified as W
import Invar.Learn.Worker.Resident qualified as Resident
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as Load
import Invar.Transcript qualified as Transcript
import LearnerFixture qualified as F
import Policies qualified
import Store (workspace)
import System.Directory (doesFileExist)
import System.Exit (ExitCode (ExitFailure))
import System.FilePath ((</>))
import Updates (alter, change, field)

learnerCalls :: Group
learnerCalls = Group "Resident learner admission" [("two acknowledged updates share a child and retain exact historical facts", once completed), ("all staged artifacts are checked before release", once artifacts), ("initial load and activation are mandatory before permission", once readiness), ("a step from another state, step or cotangent stops the update before release", once steps), ("a later update cannot replay a physical model load", once activation), ("a checked result requires one actual update measurement", once completion), ("mismatched release poisons the owner before another update", once acknowledgement), ("retired learner load instances cannot be reused", once replay), ("final closing failure propagates after acknowledged updates", once closing), ("escaped learner owners cannot start another process", once escaped), ("a process learner receives the core cotangents for every reported step", once processed), ("admission follows the sample's version and behavior policy", once admission), ("the runtime sees each attempt and step before its reply and can refuse one", once hooked), ("a resident learner asks the runtime before permission and before each reply", once residentHooks)]
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
            applied = [value | value <- F.steps exchange, F.stage value == Just "applied"]
        writeFile (W.script worker) (F.process (root </> "replies") exchange)
        outcome <- W.run worker Transcript.standard (F.call exchange)
        received <- Bytes.readFile (root </> "replies")
        pure (void outcome, received, concat [digests | value <- applied, Just digests <- [parseMaybe (withObject "applied" (.: "consumed")) value]])
    returned === Right ()
    let decoded = [reply | line <- Bytes.lines replies, Just reply <- [decodeStrict line]]
        reported = [S.digest (S.Reply 0 "" "" "" objective reward) | value <- decoded, Just (objective, reward) <- [parseMaybe (withObject "cotangents" (\fields -> (,) <$> fields .: "objective" <*> fields .: "reward")) value]]
    assert (not (null digests))
    reported === digests

admission :: PropertyT IO ()
admission = do
    root <- workspace
    let generated = L.policy F.configured
        successor = replicate 64 'd'
        stale settings = settings {L.policy = successor, L.learner = replicate 64 '9', L.schedule = L.Schedule 1 1 0 generated}
    outcomes <-
        F.admit
            root
            [ id
            , stale
            , \settings -> (stale settings) {L.schedule = L.Schedule 1 1 0 successor}
            , \settings -> (stale settings) {L.schedule = L.Schedule 1 0 1 generated}
            , \settings -> (stale settings) {L.schedule = L.Schedule 2 1 0 generated}
            , \settings -> (stale settings) {L.reference = replicate 64 '7'}
            , \settings -> settings {L.policy = successor, L.schedule = L.synchronous 0 successor}
            , \settings -> settings {L.schedule = L.Schedule 0 1 0 generated}
            , \settings -> settings {L.schedule = L.Schedule 0 1 1 generated}
            ]
    outcomes === [Right (), Right (), Left L.PolicyMismatch, Left (L.InvalidSettings "Samples of the update's own version must come from the policy being updated"), Left (L.InvalidSettings "Samples must come from version max(0, update - staleness)"), Left L.ReferenceMismatch, Left L.PolicyMismatch, Right (), Left (L.InvalidSettings "Samples must come from version max(0, update - staleness)")]

hooked :: PropertyT IO ()
hooked = do
    root <- workspace
    (accepted, refused) <- F.withAdjusted (\settings -> settings {L.steps = 2}) root $ \planned -> do
        exchange <- F.prepare root (0, binding 7) planned
        let attempt name refuse = do
                let directory = Resident.output (F.paths exchange)
                    worker = W.Worker "/bin/sh" (root </> name ++ ".sh") root "hooked checkpoint" "hooked reference" directory
                    replies = root </> name ++ ".replies"
                events <- newIORef []
                let record event = modifyIORef' events (++ [event])
                    hooks = W.Hooks (\stream -> Right () <$ record ("ready", S.identity stream, S.opening stream, 0)) $ \stream reply ->
                        if refuse == Just (S.replyStep reply)
                            then pure (Left "refused by the runtime")
                            else Right () <$ record ("reply " ++ Text.unpack (S.replySample reply), S.replyState reply, show (map Completion.step (S.completions stream)), S.replyStep reply)
                writeFile replies ""
                writeFile (W.script worker) (F.process replies exchange)
                outcome <- W.run worker Transcript.standard (W.hooked hooks (F.call exchange))
                (,,) (void outcome) <$> readIORef events <*> (length . Bytes.lines <$> Bytes.readFile replies)
        (,) <$> attempt "accepted" Nothing <*> attempt "refused" (Just 1)
    let (acceptedOutcome, acceptedEvents, acceptedReplies) = accepted
        (refusedOutcome, refusedEvents, refusedReplies) = refused
    acceptedOutcome === Right ()
    map (\(name, _, _, _) -> name) acceptedEvents === ["ready", "reply s0", "reply s1", "reply s2"]
    [(closed, index) | (_, _, closed, index) <- drop 1 acceptedEvents] === [("[]", 0), ("[]", 0), ("[0]", 1)]
    acceptedReplies === 3
    case refusedOutcome of
        Left (W.InvalidOutput (P.Refused _)) -> success
        unexpected -> annotateShow unexpected >> failure
    map (\(name, _, _, _) -> name) refusedEvents === ["ready", "reply s0", "reply s1"]
    refusedReplies === 2

residentHooks :: PropertyT IO ()
residentHooks = do
    forM_ [Nothing, Just Nothing, Just (Just 0)] $ \refusal -> do
        root <- workspace
        (returned, events) <- F.withPlan root $ \planned -> do
            original <- F.prepare root (0, binding 7) planned
            events <- newIORef []
            let record event = modifyIORef' events (++ [event])
                hooks =
                    W.Hooks
                        (\stream -> if refusal == Just Nothing then pure (Left "not ready") else Right () <$ record ("ready " ++ S.opening stream))
                        (\_ reply -> if refusal == Just (Just (S.replyStep reply)) then pure (Left "refused") else Right () <$ record ("reply " ++ Text.unpack (S.replySample reply)))
                exchange = original {F.call = W.hooked hooks (F.call original)}
            outcome <- fmap (void . fst) (F.run root (F.scenario [exchange]) (\owner -> F.execute owner [exchange]))
            (,) outcome <$> readIORef events
        case refusal of
            Nothing -> do
                returned === Right ()
                events === ["ready " ++ L.policy F.configured, "reply s0", "reply s1", "reply s2"]
            Just Nothing -> do
                refused returned
                events === []
                evalIO (doesFileExist (root </> "learner-approved0")) >>= (=== False)
            Just (Just _) -> do
                refused returned
                events === ["ready " ++ L.policy F.configured]
                evalIO (doesFileExist (root </> "learner-released0")) >>= (=== False)
  where
    refused returned = case returned of
        Left (W.InvalidOutput (P.Refused _)) -> success
        unexpected -> annotateShow unexpected >> failure
