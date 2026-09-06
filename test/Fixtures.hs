{-# LANGUAGE OverloadedStrings #-}

module Fixtures (fixtures) where

import Control.Monad (forM_)
import Data.List (find)
import Data.List.NonEmpty (NonEmpty (..))
import Data.Map.Strict qualified as Map
import Hedgehog hiding (Action)
import Invar.Spec.Request
import Model

fixtures :: Group
fixtures =
    Group
        "Request counterexamples and rules"
        [ ("stopped-guard counterexample", once stoppedGuard)
        , ("active-count counterexample", once activeCount)
        , ("bounded search finds both dependency violations", once searchMutants)
        , ("cache-ahead blockage and explicit repair", once cacheAhead)
        , ("transition effects match the specification", once effects)
        , ("transition guards reject invalid inputs", once guards)
        , ("every event enforces its status domain", once statusGuards)
        , ("initialization rejects invalid inputs", once invalidInputs)
        , ("stopping depends on generated output", once stopping)
        ]
  where
    once = withTests 1 . property

pool :: Int -> [(RequestId, Bool)] -> State
pool cap requests = require (initialize (map make requests))
  where
    make (rid, token) = (rid, require (mkCore 0 (token :| []) (fromIntegral cap)))

script :: [Action] -> [Event]
script = map (Event target)

counterexample :: Experiment -> State -> ([Event], [Event]) -> PropertyT IO String
counterexample experiment config histories@(first, second) = do
    left <- evalEither (execute (runStep experiment) config first)
    right <- evalEither (execute (runStep experiment) config second)
    let before = requireLookup target (last left)
        after = requireLookup target (last right)
        position = length (takeWhile id (zipWith (==) (generated before) (generated after)))
    let report =
            unlines
                [ "Histories: " ++ show histories
                , "Outputs: " ++ show (generated before, generated after)
                , "First divergence: decode token " ++ show position
                , "Mutation: " ++ show (mutation experiment)
                , "Changed source: " ++ changedSource (mutation experiment)
                , "Violated obligation: completed token noninterference"
                ]
    footnote report
    status before === Done
    status after === Done
    assert (generated before /= generated after)
    pure report

changedSource :: Mutation -> String
changedSource Unchanged = "unchanged execution"
changedSource NoStoppedGuard = "post-stop decode scheduling"
changedSource LeakyDecode = "active-request count"
changedSource (TruncatingPreempt _) = "prefix truncation with independent cache retention"

stoppedGuard :: PropertyT IO ()
stoppedGuard = do
    let options = Settings 255 True 1
        config = pool 2 [(target, True)]
        short = script [Admit, Chunk, Decode, Complete]
        long = script [Admit, Chunk, Decode, Decode, Complete]
    _ <- counterexample (Experiment options NoStoppedGuard) config (short, long)
    case execute (step (semantics options)) config long of
        Left (TraceError 3 (Event rid Decode) (GuardFailed rejected AlreadyStopped)) -> do
            rid === target
            rejected === target
        result -> annotateShow result >> failure

activeCount :: PropertyT IO ()
activeCount = do
    let options = Settings 0 False 1
        config = pool 1 [(target, False), (other, False)]
        first = script [Admit, Chunk, Decode, Complete]
        second = Event other Admit : first
    _ <- counterexample (Experiment options LeakyDecode) config (first, second)
    baseline <- evalEither (execute (step (semantics options)) config first)
    perturbed <- evalEither (execute (step (semantics options)) config second)
    generated (requireLookup target (last baseline)) === generated (requireLookup target (last perturbed))

cacheAhead :: PropertyT IO ()
cacheAhead = do
    let options = Settings 0 False 1
        experiment = Experiment options (TruncatingPreempt 0)
        prefix = script [Admit, Chunk, Decode, Preempt 2, Resume]
    states <- evalEither (execute (runStep experiment) (pool 2 [(target, False)]) prefix)
    let blocked = last states
        request = requireLookup target blocked
    cached request === 2
    generated request === []
    forM_ [(Chunk, CacheNotBehind), (Decode, CacheNotReady), (Complete, NotStopped)] $ \(event, reason) ->
        step (semantics options) blocked (Event target event) === Rejected (GuardFailed target reason)
    case step (semantics options) blocked (Event target Cancel) of
        Applied cancelled -> status (requireLookup target cancelled) === Cancelled
        result -> annotateShow result >> failure
    repaired <- evalEither (execute (runStep experiment) blocked (script [Preempt 0, Resume]))
    suffix <- evalEither (completion options (last repaired) target)
    finished <- evalEither (execute (runStep experiment) (last repaired) suffix)
    generated (requireLookup target (last finished)) === unroll (semantics options) (core request)

guards :: PropertyT IO ()
guards = do
    let meaning = semantics (Settings 0 False 1)
        config = pool 1 [(target, False)]
        missing = RequestId 99
    step meaning config (Event missing Admit) === Rejected (UnknownRequest missing)
    admitted <- evalEither (execute (step meaning) config (script [Admit]))
    let active = last admitted
    forM_ [(Decode, CacheNotReady), (Preempt 1, CacheIncrease), (Complete, NotStopped)] $ \(event, reason) ->
        step meaning active (Event target event) === Rejected (GuardFailed target reason)
    step meaning {frontier = \_ position -> position} active (Event target Chunk)
        === Rejected (GuardFailed target NonAdvancingFrontier)

effects :: PropertyT IO ()
effects = do
    let input = require (mkCore (2 :: Int) (False :| [True, False]) 2)
        neighbor = requireLookup other (pool 1 [(other, True)])
        state = Request input
        config request = Map.fromList [(target, request), (other, neighbor)]
        initial = config (state Pool 0 [])
        meaning =
            (semantics (Settings 0 False 1))
                { frontier = \own position -> position + fromIntegral (payload own)
                }
        expected =
            [ (Admit, state Active 0 [])
            , (Chunk, state Active 2 [])
            , (Chunk, state Active 3 [])
            , (Decode, state Active 4 [False])
            , (Preempt 4, state Paused 4 [False])
            , (Resume, state Active 4 [False])
            , (Preempt 1, state Paused 1 [False])
            , (Resume, state Active 1 [False])
            , (Chunk, state Active 3 [False])
            , (Chunk, state Active 4 [False])
            , (Decode, state Active 5 [False, False])
            , (Complete, state Done 5 [False, False])
            ]
    initialize [(target, input), (other, core neighbor)] === Right initial
    actual <- evalEither (execute (step meaning) initial (script (map fst expected)))
    actual === initial : map (config . snd) expected

invalidInputs :: PropertyT IO ()
invalidInputs = do
    mkCore (0 :: Int) (False :| []) 0 === Left ZeroLimit
    let input = core (requireLookup target (pool 1 [(target, False)]))
    initialize [(target, input), (target, input)] === Left (DuplicateRequest target)

stopping :: PropertyT IO ()
stopping = do
    let meaning = semantics (Settings 255 True 1)
        request = requireLookup target (pool 2 [(target, True)])
    assert (not (stopped meaning request))
    assert (stopped meaning request {generated = [True]})
    assert (stopped meaning request {generated = [False, False]})
    assert (not (stopped meaning request {generated = [False]}))

statusGuards :: PropertyT IO ()
statusGuards = forM_ domains $ \domain ->
    forM_ [Pool, Active, Paused, Done, Cancelled] (checkStatus domain)
  where
    domains =
        [ (Admit, [Pool])
        , (Chunk, [Active])
        , (Decode, [Active])
        , (Preempt 0, [Active])
        , (Resume, [Paused])
        , (Cancel, [Active, Paused])
        , (Complete, [Active])
        ]

checkStatus :: (Action, [Status]) -> Status -> PropertyT IO ()
checkStatus (event, allowed) state
    | state `notElem` allowed = result === Rejected (StatusMismatch target allowed state)
    | otherwise = case result of
        Applied _ -> success
        Rejected (GuardFailed rid _) -> rid === target
        unexpected -> annotateShow unexpected >> failure
  where
    meaning = semantics (Settings 0 False 1)
    request = requireLookup target (pool 2 [(target, False)])
    config = Map.singleton target request {status = state, cached = extent request}
    result = step meaning config (Event target event)

searchDepth :: Int
searchDepth = 5

searchMutants :: PropertyT IO ()
searchMutants = forM_ cases $ \(experiment@(Experiment opts _), config) -> do
    let paths = completedPaths experiment config searchDepth
    witness <- evalMaybe (mismatch paths)
    report <- counterexample experiment config witness
    evalIO (putStrLn report)
    let baseline = Experiment opts Unchanged
    mismatch (completedPaths baseline config searchDepth) === Nothing
  where
    cases =
        [ (Experiment (Settings 255 True 1) NoStoppedGuard, pool 2 [(target, True)])
        , (Experiment (Settings 0 False 1) LeakyDecode, pool 1 [(target, False), (other, False)])
        ]

completedPaths :: Experiment -> State -> Int -> [(State, [Event])]
completedPaths experiment = go
  where
    go config remaining
        | status (requireLookup target config) == Done = [(config, [])]
        | remaining == 0 = []
        | otherwise =
            [ (finished, event : rest)
            | rid <- Map.keys config
            , candidate <- [Admit, Chunk, Decode, Complete]
            , let event = Event rid candidate
            , Applied updated <- [runStep experiment config event]
            , (finished, rest) <- go updated (remaining - 1)
            ]

mismatch :: [(State, [Event])] -> Maybe ([Event], [Event])
mismatch [] = Nothing
mismatch ((config, history) : rest) = do
    (_, different) <- find ((/= observed config) . observed . fst) rest
    pure (history, different)
  where
    observed = generated . requireLookup target
