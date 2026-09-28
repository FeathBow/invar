{-# LANGUAGE OverloadedStrings #-}

module Events (events) where

import Control.Monad (forM_, unless, when)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Hedgehog hiding (Action, Command, Update)
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Async.Core (Attempt (..), Command (..), Epoch (..), Event (..), Failure (..), Phase (..), Worker (..))
import Invar.Async.Core qualified as C
import Invar.Async.Plan (Declared (..), Request (..), Update (..), Version (..))
import Invar.Async.Plan qualified as P
import Numeric.Natural (Natural)

events :: Group
events =
    Group
        "Asynchronous event core"
        [ ("a request's version is max(0, u - d) and waits for its publication", withTests 200 (property versions))
        , ("update plans reject empty, repeated, foreign and unused requests", withTests 1 (property plans))
        , ("interleavings, duplicates, lost workers, abandoned attempts and restarts reach the same decisions", withTests 400 (property interleavings))
        , ("reports from a superseded worker epoch change nothing", withTests 1 (property superseded))
        , ("an epoch at or below one a worker has used is refused after loss and recovery", withTests 1 (property regression))
        , ("only the worker epoch that started a request may complete it", withTests 1 (property ownership))
        , ("cotangents are bound to their update, step, observation and state", withTests 1 (property binding))
        , ("recovery decides publication from the commit point alone", withTests 1 (property recovery))
        , ("a different result for a completed request is a conflict", withTests 1 (property conflict))
        ]

versions :: PropertyT IO ()
versions = do
    lag <- forAll (Gen.integral (Range.linear 0 4))
    index <- forAll (Gen.integral (Range.linear 0 12))
    plan <- evalEither (P.prepare lag [Declared [Request n] [[Request n]] | n <- [0 .. index]])
    P.version plan (Update index) === Version (if index > lag then index - lag else 0)
    let Version selected = P.version plan (Update index)
    assert (P.available (Version 0) [])
    when (selected > 0) $ do
        assert (not (P.available (Version selected) [Update n | n <- [0 .. selected], n + 1 < selected]))
        assert (P.available (Version selected) [Update (selected - 1)])

plans :: PropertyT IO ()
plans = do
    P.prepare 0 [] === Left P.NoUpdates
    P.prepare 0 [Declared [] [[Request 0]]] === Left (P.EmptyUpdate (Update 0))
    P.prepare 0 [Declared [Request 0] []] === Left (P.NoSteps (Update 0))
    P.prepare 0 [Declared [Request 0] [[]]] === Left (P.EmptyStep (Update 0) 0)
    P.prepare 0 [Declared [Request 0] [[Request 1]]] === Left (P.StepOutsideUpdate (Update 0) 0 (Request 1))
    P.prepare 0 [Declared [Request 0, Request 1] [[Request 0]]] === Left (P.UnusedRequest (Update 0) (Request 1))
    P.prepare 0 [Declared [Request 0] [[Request 0]], Declared [Request 0] [[Request 0]]] === Left (P.RepeatedRequest (Request 0))
    _ <- evalEither (P.prepare 1 [Declared [Request 0, Request 1] [[Request 1], [Request 0, Request 1], [Request 0]]])
    success

declarations :: Gen (Natural, [Declared])
declarations = do
    lag <- Gen.integral (Range.linear 0 2)
    sizes <- Gen.list (Range.linear 1 4) (Gen.integral (Range.linear 1 4))
    shapes <- traverse (\size -> (,) size <$> ((,) <$> Gen.integral (Range.linear 1 size) <*> Gen.integral (Range.linear 1 2))) sizes
    let starts = scanl (+) 0 sizes
        declared = [Declared members (concat (replicate passes (chunks width members))) | (start, (size, (width, passes))) <- zip starts shapes, let members = [Request n | n <- [start .. start + size - 1]]]
    pure (lag, declared)
  where
    chunks :: Natural -> [a] -> [[a]]
    chunks _ [] = []
    chunks width values = let (taken, rest) = splitAt (fromIntegral width) values in taken : chunks width rest

data Action = Deliver Int Int | Duplicate Int | Reconnect Natural Natural | Impostor Int | Revenant Int | Abandon | Restart
    deriving (Show)

action :: Gen Action
action = Gen.frequency [(12, Deliver <$> Gen.int (Range.linear 0 20) <*> Gen.int (Range.linear 0 1)), (2, Duplicate <$> Gen.int (Range.linear 0 50)), (1, Reconnect <$> Gen.integral (Range.linear 0 1) <*> Gen.integral (Range.linear 0 2)), (1, Impostor <$> Gen.int (Range.linear 0 20)), (1, Revenant <$> Gen.int (Range.linear 0 20)), (1, pure Abandon), (1, pure Restart)]

data Item = Item {era :: Natural, work :: Work}
    deriving (Show)

data Work = Pick Request Version | Run Worker Epoch Request Version | Learn Command | Receipt Update Attempt
    deriving (Show)

data World = World
    { plan :: P.Plan
    , core :: C.State
    , pending :: [Item]
    , epochs :: Map Worker Epoch
    , issued :: Map Worker Epoch
    , bindings :: Map Request (Worker, Epoch)
    , restarts :: Natural
    , accepted :: [Event]
    , traces :: Map (Update, Attempt) [(Natural, String, String)]
    , winners :: Map Update Attempt
    , published :: [Update]
    , windows :: Natural
    , crash :: Bool
    , fuel :: Int
    }

workers :: [Worker]
workers = [Worker 0, Worker 1]

digest :: Request -> Version -> String
digest (Request request) (Version selected) = "result " ++ show request ++ " " ++ show selected

observation :: Update -> Natural -> String
observation (Update update) index = "observation " ++ show update ++ " " ++ show index

before :: Update -> Natural -> String
before (Update update) index = "state " ++ show update ++ " " ++ show index

count :: P.Plan -> Update -> Natural
count chosen update = maybe 0 (fromIntegral . length . steps) (P.declared chosen update)

interleavings :: PropertyT IO ()
interleavings = do
    (lag, declared) <- forAll declarations
    chosen <- evalEither (P.prepare lag declared)
    actions <- forAll (Gen.list (Range.linear 0 80) action)
    crash <- forAll Gen.bool
    let (initial, commands) = C.start chosen
        world = World chosen initial [] (Map.fromList [(worker, Epoch 0) | worker <- workers]) (Map.fromList [(worker, Epoch 0) | worker <- workers]) Map.empty 0 [] Map.empty Map.empty [] 0 False 20000
    connected <- foldM' world [Connected worker (Epoch 0) | worker <- workers]
    performed <- perform (enqueue connected commands) actions
    finished <- drain performed {crash = crash}
    cover 20 "a restart fell between a commit and its receipt" (windows finished > 0)
    C.committed (core finished) === P.updates chosen
    C.results (core finished) === Map.fromList [(request, digest request (P.version chosen update)) | update <- P.updates chosen, Just (Declared members _) <- [P.declared chosen update], request <- members]
    forM_ (P.updates chosen) $ \update -> do
        attempt <- maybe failure pure (Map.lookup update (winners finished))
        Map.lookup (update, attempt) (traces finished) === Just [(index, observation update index, before update index) | index <- [0 .. count chosen update - 1]]
  where
    foldM' world [] = pure world
    foldM' world (event : rest) = feed world 0 event >>= (`foldM'` rest)

enqueue :: World -> [Command] -> World
enqueue world commands = world {pending = pending world ++ [Item (restarts world) (toWork command) | command <- commands]}
  where
    toWork (Dispatch request selected) = Pick request selected
    toWork command = Learn command

perform :: World -> [Action] -> PropertyT IO World
perform world [] = pure world
perform world (next : rest) = do
    when (fuel world <= 0) (annotate "out of fuel" >> failure)
    updated <- case next of
        Deliver index choice | not (null (pending world)) -> deliver world (index `mod` length (pending world)) choice
        Deliver _ _ -> pure world
        Duplicate index | not (null repeatable) -> do
            let event = repeatable !! (index `mod` length repeatable)
            case C.step (core world) event of
                Right (after, commands) -> after === core world >> commands === []
                Left problem -> unless (stale world (restarts world) event && rejection problem) (annotateShow (event, problem) >> failure)
            pure world
          where
            repeatable = [event | event <- accepted world, duplicable event]
        Duplicate _ -> pure world
        Reconnect index back -> do
            let worker = workers !! fromIntegral (index `mod` 2)
                Epoch highest = issued world Map.! worker
                reused = Epoch (if highest >= back then highest - back else 0)
            lost <- feed world (restarts world) (Lost worker (epochs world Map.! worker))
            C.step (core lost) (Connected worker reused) === Left (StaleEpoch worker reused)
            let fresh = Epoch (highest + 1)
            feed lost {epochs = Map.insert worker fresh (epochs lost), issued = Map.insert worker fresh (issued lost)} (restarts lost) (Connected worker fresh)
        Impostor index -> case [(other, request, selected) | Item _ (Run worker _ request selected) <- pending world, let other = if worker == Worker 0 then Worker 1 else Worker 0, Map.lookup request (bindings world) /= Just (other, epochs world Map.! other)] of
            [] -> pure world
            candidates -> do
                let (other, request, selected) = candidates !! (index `mod` length candidates)
                    event = Completed other (epochs world Map.! other) request (digest request selected)
                case C.step (core world) event of
                    Left problem | rejection problem -> pure world
                    outcome -> annotateShow (event, outcome) >> annotate "a completion from a worker that did not start the request was accepted" >> failure
        Revenant index -> case [(worker, epoch, request, selected) | Item _ (Run worker epoch request selected) <- pending world, Map.lookup worker (epochs world) /= Just epoch] of
            [] -> pure world
            candidates -> do
                let (worker, epoch, request, selected) = candidates !! (index `mod` length candidates)
                started <- feed world (restarts world) (Started worker epoch request)
                feed started (restarts started) (Completed worker epoch request (digest request selected))
        Abandon -> case C.learning (core world) of
            Just (update, phase) | not (committing phase) -> feed world (restarts world) (Abandoned update (attemptOf phase))
            _ -> pure world
        Restart -> do
            let window = [() | Item issued (Receipt _ _) <- pending world, issued == restarts world]
            let base = 1000 * (restarts world + 1)
            (recovered, commands) <- evalEither (C.recover (plan world) (published world) (C.results (core world)) (issued world) base)
            let raised = Map.map (\(Epoch epoch) -> Epoch (epoch + 1)) (issued world)
                restarted = enqueue world {core = recovered, restarts = restarts world + 1, windows = windows world + (if null window then 0 else 1), epochs = raised, issued = raised, bindings = Map.empty} commands
            foldl (\acted (worker, epoch) -> acted >>= \current -> feed current (restarts current) (Connected worker epoch)) (pure restarted) (Map.toList (epochs restarted))
    perform updated {fuel = fuel updated - 1} rest

drain :: World -> PropertyT IO World
drain world
    | C.committed (core world) == P.updates (plan world) = pure world
    | null (pending world) = annotateShow (C.learning (core world)) >> annotate "stuck with nothing pending" >> failure
    | fuel world <= 0 = annotate "out of fuel" >> failure
    | crash world, Item issued (Receipt _ _) : _ <- pending world, issued == restarts world = perform world {crash = False} [Restart] >>= drain
    | otherwise = deliver world 0 0 >>= \next -> drain next {fuel = fuel next - 1}

deliver :: World -> Int -> Int -> PropertyT IO World
deliver world index choice = do
    (item, remaining) <- case splitAt index (pending world) of
        (earlier, chosen : later) -> pure (chosen, earlier ++ later)
        _ -> failure
    let rest = world {pending = remaining}
    case work item of
        _ | era item /= restarts world && not (running (work item)) -> pure rest
        Pick request selected -> do
            let worker = workers !! (choice `mod` 2)
                epoch = epochs world Map.! worker
            started <- feed rest (era item) (Started worker epoch request)
            pure started {pending = pending started ++ [Item (era item) (Run worker epoch request selected)]}
        Run worker epoch request selected -> feed rest (era item) (Completed worker epoch request (digest request selected))
        Learn command -> learn rest (era item) command
        Receipt update attempt -> feed rest (era item) (Committed update attempt)

learn :: World -> Natural -> Command -> PropertyT IO World
learn world issued command = case command of
    Send update attempt -> do
        proximal <- feed world issued (Proximal update attempt (observation update 0) (before update 0))
        feed proximal issued (Current update attempt 0 (observation update 0) (before update 0))
    Cotangents update attempt index seen state -> do
        let traced = world {traces = Map.insertWith (flip (++)) (update, attempt) [(index, seen, state)] (traces world)}
        applied <- feed traced issued (Applied update attempt index seen (before update (index + 1)))
        if index + 1 < count (plan world) update
            then feed applied issued (Current update attempt (index + 1) (observation update (index + 1)) (before update (index + 1)))
            else feed applied issued (Staged update attempt ("successor " ++ show update))
    Record update attempt _ -> feed world issued (Recorded update attempt)
    Commit update attempt -> do
        when (update `elem` published world) (annotateShow update >> annotate "a published update was committed again" >> failure)
        pure world {published = published world ++ [update], winners = Map.insert update attempt (winners world), pending = pending world ++ [Item issued (Receipt update attempt)]}
    Dispatch _ _ -> failure

feed :: World -> Natural -> Event -> PropertyT IO World
feed world issued event = case C.step (core world) event of
    Right (next, commands) -> do
        when (stale world issued event) (annotateShow event >> annotate "a stale event was accepted" >> failure)
        forM_ commands (checked world)
        pure (enqueue world {core = next, accepted = event : accepted world, bindings = bound world event} commands)
    Left problem -> do
        unless (stale world issued event && rejection problem) (annotateShow (event, problem) >> failure)
        pure world

checked :: World -> Command -> PropertyT IO ()
checked world command = case command of
    Dispatch request selected -> do
        update <- maybe failure pure (P.owner (plan world) request)
        selected === P.version (plan world) update
    Send update _ -> assert (update `notElem` published world)
    _ -> pure ()

stale :: World -> Natural -> Event -> Bool
stale world issued event =
    issued /= restarts world || case event of
        Started worker epoch request -> Map.lookup worker (epochs world) /= Just epoch || maybe False (/= (worker, epoch)) (Map.lookup request (bindings world)) && Map.notMember request (C.results (core world))
        Completed worker epoch request _ -> Map.lookup worker (epochs world) /= Just epoch || Map.lookup request (bindings world) /= Just (worker, epoch)
        _ -> case (addressed event, C.learning (core world)) of
            (Just (update, attempt), Just (active, phase)) -> update /= active || attempt /= attemptOf phase
            (Just _, Nothing) -> True
            _ -> False

bound :: World -> Event -> Map Request (Worker, Epoch)
bound world event = case event of
    Started worker epoch request
        | Map.notMember request (C.results (core world)) -> Map.insertWith (\_ existing -> existing) request (worker, epoch) (bindings world)
    Lost worker _ -> released worker
    Connected worker _ -> released worker
    _ -> bindings world
  where
    released worker = Map.filterWithKey (\request (holder, _) -> holder /= worker || Map.member request (C.results (core world))) (bindings world)

rejection :: Failure -> Bool
rejection problem = case problem of
    StaleEpoch _ _ -> True
    NotConnected _ _ -> True
    StaleAttempt _ _ -> True
    Occupied _ -> True
    NotRunning {} -> True
    Unexpected _ -> True
    _ -> False

duplicable :: Event -> Bool
duplicable event = case event of
    Completed {} -> True
    Connected {} -> False
    Lost {} -> False
    Started {} -> False
    _ -> True

addressed :: Event -> Maybe (Update, Attempt)
addressed event = case event of
    Proximal update attempt _ _ -> Just (update, attempt)
    Current update attempt _ _ _ -> Just (update, attempt)
    Applied update attempt _ _ _ -> Just (update, attempt)
    Staged update attempt _ -> Just (update, attempt)
    Recorded update attempt -> Just (update, attempt)
    Committed update attempt -> Just (update, attempt)
    Abandoned update attempt -> Just (update, attempt)
    _ -> Nothing

attemptOf :: Phase -> Attempt
attemptOf phase = case phase of
    Scoring attempt -> attempt
    Forward attempt _ _ -> attempt
    Backward attempt _ _ _ -> attempt
    Staging attempt -> attempt
    Recording attempt _ -> attempt
    Committing attempt -> attempt

running :: Work -> Bool
running Run {} = True
running _ = False

committing :: Phase -> Bool
committing (Committing _) = True
committing _ = False

single :: IO (P.Plan, C.State)
single = do
    chosen <- either (fail . show) pure (P.prepare 0 [Declared [Request 0] [[Request 0], [Request 0]]])
    let (initial, _) = C.start chosen
    either (fail . show) (pure . (,) chosen . fst) (C.step initial (Connected (Worker 0) (Epoch 0)))

run :: C.State -> [Event] -> Either Failure C.State
run = foldl (\state event -> state >>= fmap fst . (`C.step` event)) . Right

superseded :: PropertyT IO ()
superseded = do
    (_, connected) <- evalIO single
    started <- evalEither (run connected [Started (Worker 0) (Epoch 0) (Request 0), Lost (Worker 0) (Epoch 0), Connected (Worker 0) (Epoch 1)])
    C.step started (Completed (Worker 0) (Epoch 0) (Request 0) "late") === Left (StaleEpoch (Worker 0) (Epoch 0))
    C.step started (Connected (Worker 0) (Epoch 1)) === Left (StaleEpoch (Worker 0) (Epoch 1))
    C.step started (Completed (Worker 1) (Epoch 0) (Request 0) "unknown") === Left (NotConnected (Worker 1) (Epoch 0))
    C.results started === Map.empty

binding :: PropertyT IO ()
binding = do
    (_, connected) <- evalIO single
    let update = Update 0
        attempt = Attempt 0
    scoring <- evalEither (run connected [Started (Worker 0) (Epoch 0) (Request 0), Completed (Worker 0) (Epoch 0) (Request 0) "r", Proximal update attempt "p" "s0"])
    C.step scoring (Current update attempt 0 "o0" "other") === Left (BindingMismatch update attempt 0)
    C.step scoring (Current update attempt 1 "o0" "s0") === Left (BindingMismatch update attempt 1)
    (forward, commands) <- evalEither (C.step scoring (Current update attempt 0 "o0" "s0"))
    commands === [Cotangents update attempt 0 "o0" "s0"]
    C.step forward (Applied update attempt 0 "late" "s1") === Left (BindingMismatch update attempt 0)
    C.step forward (Applied update attempt 1 "o0" "s1") === Left (BindingMismatch update attempt 1)
    C.step forward (Applied update (Attempt 7) 0 "o0" "s1") === Left (StaleAttempt update (Attempt 7))
    second <- evalEither (run forward [Applied update attempt 0 "o0" "s1"])
    C.step second (Current update attempt 1 "o1" "s0") === Left (BindingMismatch update attempt 1)
    (_, next) <- evalEither (C.step second (Current update attempt 1 "o1" "s1"))
    next === [Cotangents update attempt 1 "o1" "s1"]

recovery :: PropertyT IO ()
recovery = do
    chosen <- evalEither (P.prepare 0 [Declared [Request 0] [[Request 0]], Declared [Request 1] [[Request 1]]])
    let stored = Map.fromList [(Request 0, "r0")]
    (unpublished, commands) <- evalEither (C.recover chosen [] stored Map.empty 5)
    commands === [Send (Update 0) (Attempt 5)]
    (published, resumed) <- evalEither (C.recover chosen [Update 0] stored Map.empty 5)
    resumed === [Dispatch (Request 1) (Version 1)]
    C.committed published === [Update 0]
    C.recover chosen [Update 1] stored Map.empty 5 === Left InvalidRecovery
    recording <- evalEither (run unpublished [Proximal (Update 0) (Attempt 5) "p" "s0", Current (Update 0) (Attempt 5) 0 "o" "s0", Applied (Update 0) (Attempt 5) 0 "o" "s1", Staged (Update 0) (Attempt 5) "d", Recorded (Update 0) (Attempt 5)])
    C.learning recording === Just (Update 0, Committing (Attempt 5))
    C.step recording (Abandoned (Update 0) (Attempt 5)) === Left (UncertainCommit (Update 0) (Attempt 5))

conflict :: PropertyT IO ()
conflict = do
    (_, connected) <- evalIO single
    completed <- evalEither (run connected [Started (Worker 0) (Epoch 0) (Request 0), Completed (Worker 0) (Epoch 0) (Request 0) "first"])
    (same, commands) <- evalEither (C.step completed (Completed (Worker 0) (Epoch 0) (Request 0) "first"))
    same === completed
    commands === []
    C.step completed (Completed (Worker 0) (Epoch 0) (Request 0) "second") === Left (Conflict (Request 0))

regression :: PropertyT IO ()
regression = do
    (chosen, connected) <- evalIO single
    raised <- evalEither (run connected [Lost (Worker 0) (Epoch 0), Connected (Worker 0) (Epoch 5), Lost (Worker 0) (Epoch 5)])
    C.step raised (Connected (Worker 0) (Epoch 1)) === Left (StaleEpoch (Worker 0) (Epoch 1))
    C.step raised (Connected (Worker 0) (Epoch 5)) === Left (StaleEpoch (Worker 0) (Epoch 5))
    (recovered, _) <- evalEither (C.recover chosen [] Map.empty (C.highest raised) 9)
    C.step recovered (Connected (Worker 0) (Epoch 1)) === Left (StaleEpoch (Worker 0) (Epoch 1))
    _ <- evalEither (C.step recovered (Connected (Worker 0) (Epoch 6)))
    success

ownership :: PropertyT IO ()
ownership = do
    (_, connected) <- evalIO single
    started <- evalEither (run connected [Connected (Worker 1) (Epoch 0), Started (Worker 0) (Epoch 0) (Request 0)])
    C.step started (Completed (Worker 1) (Epoch 0) (Request 0) "impostor") === Left (NotRunning (Request 0) (Worker 1) (Epoch 0))
    C.step started (Started (Worker 1) (Epoch 0) (Request 0)) === Left (Occupied (Request 0))
    (same, none) <- evalEither (C.step started (Started (Worker 0) (Epoch 0) (Request 0)))
    same === started
    none === []
    completed <- evalEither (run started [Completed (Worker 0) (Epoch 0) (Request 0) "r"])
    C.step completed (Completed (Worker 1) (Epoch 0) (Request 0) "r") === Left (NotRunning (Request 0) (Worker 1) (Epoch 0))
