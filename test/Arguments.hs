{-# LANGUAGE OverloadedStrings #-}

module Arguments (arguments) where

import Control.Monad (forM_)
import Data.List.NonEmpty qualified as NonEmpty
import Data.Map.Strict qualified as Map
import Hedgehog
import History
import Invar.Spec.Request
import Model
import Properties (campaign)

data Call = Call (Core Int Bool) [Bool]
    deriving (Eq, Show)

data Observed = Observed {value :: Bool, call :: Maybe Call}
    deriving (Eq, Show)

arguments :: Group
arguments =
    Group
        "Decoder arguments"
        [("actual arguments are request-owned proper reference prefixes", campaign actualArguments)]

actualArguments :: PropertyT IO ()
actualArguments = do
    pair <- forAll genPair
    let meaning = instrument (options pair)
        config = Map.map observeRequest (initial pair)
    forM_ [leftHistory pair, rightHistory pair] $ \history -> do
        states <- evalEither (execute (step meaning) config history)
        forM_ states (checkState pair)

checkState :: Pair -> Config Int Observed -> PropertyT IO ()
checkState pair state = forM_ (Map.toList state) $ \(rid, request) ->
    checkArguments (options pair) (core (requireLookup rid (initial pair))) (generated request)

checkArguments :: Settings -> Core Int Bool -> [Observed] -> PropertyT IO ()
checkArguments opts input tokens = forM_ (zip [0 ..] tokens) $ \(index, token) -> do
    let reference = unroll (semantics opts) input
        prefix = NonEmpty.toList (prompt input)
    assert (index < length reference)
    call token === Just (Call input (prefix ++ take index reference))
    value token === reference !! index

instrument :: Settings -> Semantics Int Observed
instrument opts =
    Semantics
        { next = \input tokens ->
            let original = plainCore input
                observedContext = map value tokens
             in Observed
                    (oracle opts original observedContext)
                    (Just (Call original observedContext))
        , frontier = \_ position -> position + stride opts
        , terminal = \token -> stopOnTrue opts && value token
        }

plainCore :: Core Int Observed -> Core Int Bool
plainCore input = require (mkCore (payload input) (fmap value (prompt input)) (limit input))

observeRequest :: Request Int Bool -> Request Int Observed
observeRequest request =
    Request
        { core = require (mkCore (payload input) (fmap plain (prompt input)) (limit input))
        , status = status request
        , cached = cached request
        , generated = map plain (generated request)
        }
  where
    input = core request
    plain token = Observed token Nothing
