{-# LANGUAGE OverloadedStrings #-}

module Properties (properties, campaign) where

import Control.Monad (forM_)
import Data.List (isPrefixOf)
import Data.Map.Strict qualified as Map
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import History
import Invar.Spec.Request
import Model

campaignTests :: TestLimit
campaignTests = 300

firstDiscard :: DiscardLimit
firstDiscard = 1

campaign :: PropertyT IO () -> Property
campaign = withTests campaignTests . withDiscards firstDiscard . property

coverHistory :: LabelName -> Bool -> PropertyT IO ()
coverHistory = cover minimumPercent
  where
    minimumPercent = 20

properties :: Group
properties =
    Group
        "Request histories"
        [ ("legal and equivalent history pairs", campaign pairedHistories)
        , ("all visited prefixes follow the reference", campaign prefixes)
        , ("completed histories agree with the reference", campaign completed)
        , ("truncating preemption preserves token invariance", campaign truncating)
        ]

pairRuns :: Pair -> PropertyT IO ([State], [State])
pairRuns pair = do
    let transition = step (semantics (options pair))
    first <- evalEither (execute transition (initial pair) (leftHistory pair))
    second <- evalEither (execute transition (initial pair) (rightHistory pair))
    pure (first, second)

pairedHistories :: PropertyT IO ()
pairedHistories = do
    pair <- forAll genPair
    (first, second) <- pairRuns pair
    assert (leftHistory pair /= rightHistory pair)
    forM_ (first ++ second) $ \state ->
        Map.map core state === Map.map core (initial pair)
    status (requireLookup target (last first)) === Done
    status (requireLookup target (last second)) === Done
    let events = rightHistory pair
    coverHistory "cancellation" (any ((== Cancel) . action) events)
    coverHistory "other request completes" (status (requireLookup other (last second)) == Done)
    coverHistory "cache lowered" (cacheLowered second events)
    coverHistory "cache preserved" (cachePreserved second events)
    forM_ [Admit, Chunk, Decode, Resume, Complete] $ \event ->
        assert (any ((== event) . action) events)

assertPrefixes :: Settings -> [State] -> PropertyT IO ()
assertPrefixes opts states = forM_ states $ \state ->
    forM_ (Map.elems state) $ \request ->
        assert (generated request `isPrefixOf` unroll (semantics opts) (core request))

prefixes :: PropertyT IO ()
prefixes = do
    pair <- forAll genPair
    (first, second) <- pairRuns pair
    config <- forAll (Gen.element (filter hasPrefix second))
    paused <- forAll Gen.bool
    let request = requireLookup target config
        pause = [Event target (Preempt (cached request)) | paused]
        events = pause ++ [Event target Cancel]
    cancelled <- evalEither (execute (step (semantics (options pair))) config events)
    assertPrefixes (options pair) (first ++ second ++ cancelled)
    status (requireLookup target (last cancelled)) === Cancelled
    generated (requireLookup target (last cancelled)) === generated request
    coverHistory "cancel active prefix" (not paused)
    coverHistory "cancel paused prefix" paused

completed :: PropertyT IO ()
completed = do
    pair <- forAll genPair
    (first, second) <- pairRuns pair
    let request = requireLookup target (last first)
        replayed = requireLookup target (last second)
    generated request === generated replayed
    generated request === unroll (semantics (options pair)) (core request)

truncating :: PropertyT IO ()
truncating = do
    pair <- forAll genPair
    (states, _) <- pairRuns pair
    config <- forAll (Gen.element (filter hasPrefix states))
    let request = requireLookup target config
        size = length (generated request)
    keep <- forAll (Gen.int (Range.constant 0 size))
    let experiment = Experiment (options pair) (TruncatingPreempt (fromIntegral keep))
        transitions = [Event target (Preempt 0), Event target Resume]
    mutated <- evalEither (execute (runStep experiment) config transitions)
    suffix <- evalEither (completion (options pair) (last mutated) target)
    finished <- evalEither (execute (runStep experiment) (last mutated) suffix)
    assertPrefixes (options pair) (mutated ++ finished)
    generated (requireLookup target (last finished)) === unroll (semantics (options pair)) (core request)
    coverHistory "nonempty prefix truncated" (keep < size)

hasPrefix :: State -> Bool
hasPrefix state =
    let request = requireLookup target state
     in status request == Active && not (null (generated request))
