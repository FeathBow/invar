{-# LANGUAGE OverloadedStrings #-}

module Schedules (schedules) where

import Control.Monad (forM_)
import Data.List (permutations)
import Hedgehog
import Invar.Schedule qualified as S
import Numeric.Natural (Natural)

schedules :: Group
schedules = Group "Independent history axes" [("execution and delivery preserve member ownership independently", once independent), ("each axis rejects incomplete repeated and foreign positions", once invalid), ("missing or relabeled results cannot be delivered", once membership)]
  where
    once = withTests 1 . property

positions :: [Natural]
positions = [0, 1, 2]

memberCount :: Natural
memberCount = fromIntegral (length positions)

independent :: PropertyT IO ()
independent = forM_ (permutations positions) $ \execution ->
    forM_ (permutations positions) $ \delivery -> do
        plan <- evalEither (S.prepare memberCount execution delivery)
        started <- evalEither (S.execute plan (map show positions))
        started === [(position, show position) | position <- execution]
        arrived <- evalEither (S.deliver plan started)
        arrived === [(position, show position) | position <- delivery]

invalid :: PropertyT IO ()
invalid = forM_ [[], [0], [0, 1, 1], [0, 1, 3], positions ++ [memberCount]] $ \changed -> do
    rejected (S.InvalidPermutation S.Execution) (S.prepare memberCount changed positions)
    rejected (S.InvalidPermutation S.Delivery) (S.prepare memberCount positions changed)

membership :: PropertyT IO ()
membership = do
    plan <- evalEither (S.prepare memberCount (reverse positions) positions)
    let members = map show positions
    forM_ [[], take 1 members, members ++ ["foreign"]] $ \values ->
        rejected S.MemberMismatch (S.execute plan values)
    started <- evalEither (S.execute plan members)
    forM_ [[], take 1 started, started ++ take 1 started, (memberCount, "foreign") : drop 1 started] $ \values ->
        rejected S.MemberMismatch (S.deliver plan values)

rejected :: S.Error -> Either S.Error value -> PropertyT IO ()
rejected expected result = case result of
    Left actual -> actual === expected
    Right _ -> failure
