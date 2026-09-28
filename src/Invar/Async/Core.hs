{-# LANGUAGE Safe #-}

module Invar.Async.Core (Worker (..), Epoch (..), Attempt (..), Digest, Event (..), Command (..), Failure (..), Phase (..), State, start, step, recover, committed, results, learning, highest) where

import Control.Monad (unless)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (isJust, isNothing)
import Data.Set (Set)
import Data.Set qualified as Set
import Invar.Async.Plan (Declared (..), Plan, Request, Update (..), Version, available, declared, owner, updates, version)
import Numeric.Natural (Natural)

newtype Worker = Worker Natural
    deriving (Eq, Ord, Show)

newtype Epoch = Epoch Natural
    deriving (Eq, Ord, Show)

newtype Attempt = Attempt Natural
    deriving (Eq, Ord, Show)

type Digest = String

data Event
    = Connected Worker Epoch
    | Lost Worker Epoch
    | Started Worker Epoch Request
    | Completed Worker Epoch Request Digest
    | Proximal Update Attempt Digest Digest
    | Current Update Attempt Natural Digest Digest
    | Applied Update Attempt Natural Digest Digest
    | Staged Update Attempt Digest
    | Recorded Update Attempt
    | Committed Update Attempt
    | Abandoned Update Attempt
    deriving (Eq, Ord, Show)

data Command
    = Dispatch Request Version
    | Send Update Attempt
    | Cotangents Update Attempt Natural Digest Digest
    | Record Update Attempt Digest
    | Commit Update Attempt
    deriving (Eq, Ord, Show)

data Failure
    = UnknownRequest Request
    | NotDispatched Request
    | NotConnected Worker Epoch
    | StaleEpoch Worker Epoch
    | Conflict Request
    | Occupied Request
    | NotRunning Request Worker Epoch
    | StaleAttempt Update Attempt
    | Unexpected Event
    | BindingMismatch Update Attempt Natural
    | UncertainCommit Update Attempt
    | InvalidRecovery
    deriving (Eq, Show)

data Phase
    = Scoring Attempt
    | Forward Attempt Natural Digest
    | Backward Attempt Natural Digest Digest
    | Staging Attempt
    | Recording Attempt Digest
    | Committing Attempt
    deriving (Eq, Show)

data State = State
    { plan :: Plan
    , epochs :: Map Worker Epoch
    , used :: Map Worker Epoch
    , dispatched :: Set Request
    , running :: Map Request (Worker, Epoch)
    , completedResults :: Map Request Digest
    , completers :: Map Request (Worker, Epoch)
    , done :: [Update]
    , active :: Maybe (Update, Phase)
    , attempts :: Natural
    , seen :: Set Event
    }
    deriving (Eq, Show)

committed :: State -> [Update]
committed = done

results :: State -> Map Request Digest
results = completedResults

learning :: State -> Maybe (Update, Phase)
learning = active

highest :: State -> Map Worker Epoch
highest = used

start :: Plan -> (State, [Command])
start chosen = advance (State chosen Map.empty Map.empty Set.empty Map.empty Map.empty Map.empty [] Nothing 0 Set.empty)

recover :: Plan -> [Update] -> Map Request Digest -> Map Worker Epoch -> Natural -> Either Failure (State, [Command])
recover chosen published stored lowest fresh = do
    unless (published == take (length published) (updates chosen)) (Left InvalidRecovery)
    unless (all (isJust . owner chosen) (Map.keys stored)) (Left InvalidRecovery)
    pure (advance (State chosen Map.empty lowest (Map.keysSet stored) Map.empty stored Map.empty published Nothing fresh Set.empty))

step :: State -> Event -> Either Failure (State, [Command])
step state event = case event of
    Connected worker epoch -> do
        case Map.lookup worker (used state) of
            Just previous | epoch <= previous -> Left (StaleEpoch worker epoch)
            _ -> pure ()
        let (released, kept) = Map.partition ((== worker) . fst) (running state)
        pure (redispatch (Map.keys released) state {epochs = Map.insert worker epoch (epochs state), used = Map.insert worker epoch (used state), running = kept})
    Lost worker epoch -> do
        current state worker epoch
        let (released, kept) = Map.partition (== (worker, epoch)) (running state)
        pure (redispatch (Map.keys released) state {epochs = Map.delete worker (epochs state), running = kept})
    Started worker epoch request -> do
        current state worker epoch
        known state request
        case Map.lookup request (running state) of
            _ | Map.member request (completedResults state) -> pure (state, [])
            Just owner'
                | owner' == (worker, epoch) -> pure (state, [])
                | otherwise -> Left (Occupied request)
            Nothing -> pure (state {running = Map.insert request (worker, epoch) (running state)}, [])
    Completed worker epoch request digest -> do
        current state worker epoch
        known state request
        let reporter = (worker, epoch)
        case (Map.lookup request (completedResults state), Map.lookup request (completers state)) of
            (Just previous, Just completer)
                | completer /= reporter -> Left (NotRunning request worker epoch)
                | previous == digest -> pure (state, [])
                | otherwise -> Left (Conflict request)
            (Just _, Nothing) -> Left (NotRunning request worker epoch)
            (Nothing, _)
                | Map.lookup request (running state) /= Just reporter -> Left (NotRunning request worker epoch)
                | otherwise -> pure (advance state {completedResults = Map.insert request digest (completedResults state), completers = Map.insert request reporter (completers state), running = Map.delete request (running state)})
    _ | Set.member event (seen state) -> pure (state, [])
    _ -> learner state event

current :: State -> Worker -> Epoch -> Either Failure ()
current state worker epoch = case Map.lookup worker (epochs state) of
    Just connected
        | connected == epoch -> pure ()
        | epoch < connected -> Left (StaleEpoch worker epoch)
    _ -> Left (NotConnected worker epoch)

learner :: State -> Event -> Either Failure (State, [Command])
learner state event = do
    (update, phase) <- maybe (Left (Unexpected event)) Right (active state)
    (target, attempt) <- maybe (Left (Unexpected event)) Right (addressed event)
    unless (target == update && attempt == attemptOf phase) (Left (StaleAttempt target attempt))
    count <- maybe (Left (Unexpected event)) (Right . fromIntegral . length . steps) (declared (plan state) update)
    let accepted next commands = pure (state {active = next, seen = Set.insert event (seen state)}, commands)
        moving next = accepted (Just (update, next))
    case (event, phase) of
        (Proximal _ _ _ before, Scoring _) -> moving (Forward attempt 0 before) []
        (Current _ _ index observation before, Forward _ expected state')
            | index == expected && before == state' -> moving (Backward attempt index observation before) [Cotangents update attempt index observation before]
            | otherwise -> Left (BindingMismatch update attempt index)
        (Applied _ _ index observation after, Backward _ expected bound _)
            | index == expected && observation == bound -> moving (if index + 1 < count then Forward attempt (index + 1) after else Staging attempt) []
            | otherwise -> Left (BindingMismatch update attempt index)
        (Staged _ _ digest, Staging _) -> moving (Recording attempt digest) [Record update attempt digest]
        (Recorded _ _, Recording _ _) -> moving (Committing attempt) [Commit update attempt]
        (Committed _ _, Committing _) -> do
            let (next, commands) = advance state {active = Nothing, done = done state ++ [update], seen = Set.insert event (seen state)}
            pure (next, commands)
        (Abandoned _ _, Committing _) -> Left (UncertainCommit update attempt)
        (Abandoned _ _, _) -> do
            let fresh = Attempt (attempts state)
            pure (state {active = Just (update, Scoring fresh), attempts = attempts state + 1, seen = Set.insert event (seen state)}, [Send update fresh])
        _ -> Left (Unexpected event)

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

redispatch :: [Request] -> State -> (State, [Command])
redispatch released state = (state, [Dispatch request (version (plan state) update) | request <- released, Just update <- [owner (plan state) request]])

advance :: State -> (State, [Command])
advance state = (next, dispatches ++ starting)
  where
    ready =
        [ (request, selected)
        | update <- updates (plan state)
        , update `notElem` done state
        , let selected = version (plan state) update
        , available selected (done state)
        , Just (Declared requests _) <- [declared (plan state) update]
        , request <- requests
        , Set.notMember request (dispatched state)
        ]
    dispatches = [Dispatch request selected | (request, selected) <- ready]
    marked = state {dispatched = foldr (Set.insert . fst) (dispatched state) ready}
    upcoming = Update (fromIntegral (length (done state)))
    complete = case declared (plan state) upcoming of
        Just (Declared requests _) -> all (`Map.member` completedResults state) requests
        Nothing -> False
    (next, starting)
        | isNothing (active state) && complete =
            let fresh = Attempt (attempts marked)
             in (marked {active = Just (upcoming, Scoring fresh), attempts = attempts marked + 1}, [Send upcoming fresh])
        | otherwise = (marked, [])

known :: State -> Request -> Either Failure ()
known state request = do
    unless (isJust (owner (plan state) request)) (Left (UnknownRequest request))
    unless (Set.member request (dispatched state)) (Left (NotDispatched request))
