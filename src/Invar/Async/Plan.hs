{-# LANGUAGE Safe #-}

module Invar.Async.Plan (Request (..), Update (..), Version (..), Declared (..), Plan, Error (..), prepare, staleness, updates, declared, owner, version, available) where

import Control.Monad (forM_, unless, when)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Numeric.Natural (Natural)

newtype Request = Request Natural
    deriving (Eq, Ord, Show)

newtype Update = Update Natural
    deriving (Eq, Ord, Show)

newtype Version = Version Natural
    deriving (Eq, Ord, Show)

data Declared = Declared {members :: [Request], steps :: [[Request]]}
    deriving (Eq, Show)

data Plan = Plan {staleness :: Natural, declaredUpdates :: [Declared], owners :: Map Request Update}
    deriving (Eq, Show)

data Error
    = NoUpdates
    | EmptyUpdate Update
    | NoSteps Update
    | RepeatedRequest Request
    | EmptyStep Update Natural
    | StepOutsideUpdate Update Natural Request
    | UnusedRequest Update Request
    deriving (Eq, Show)

prepare :: Natural -> [Declared] -> Either Error Plan
prepare lag planned = do
    when (null planned) (Left NoUpdates)
    let indexed = zip (map Update [0 ..]) planned
        everyone = concatMap members planned
    forM_ indexed $ \(update, Declared requests batches) -> do
        when (null requests) (Left (EmptyUpdate update))
        when (null batches) (Left (NoSteps update))
        forM_ (zip [0 ..] batches) $ \(index, batch) -> do
            when (null batch) (Left (EmptyStep update index))
            forM_ batch $ \request -> unless (request `elem` requests) (Left (StepOutsideUpdate update index request))
        forM_ requests $ \request -> unless (any (request `elem`) batches) (Left (UnusedRequest update request))
    case [request | (request, count) <- Map.toList (Map.fromListWith (+) [(request, 1 :: Natural) | request <- everyone]), count > 1] of
        request : _ -> Left (RepeatedRequest request)
        [] -> pure ()
    pure (Plan lag planned (Map.fromList [(request, update) | (update, Declared requests _) <- indexed, request <- requests]))

updates :: Plan -> [Update]
updates plan = zipWith const (map Update [0 ..]) (declaredUpdates plan)

declared :: Plan -> Update -> Maybe Declared
declared plan (Update index) = case drop (fromIntegral index) (declaredUpdates plan) of
    selected : _ -> Just selected
    [] -> Nothing

owner :: Plan -> Request -> Maybe Update
owner plan request = Map.lookup request (owners plan)

version :: Plan -> Update -> Version
version plan (Update index) = Version (if index > staleness plan then index - staleness plan else 0)

available :: Version -> [Update] -> Bool
available (Version 0) _ = True
available (Version index) committed = Update (index - 1) `elem` committed
