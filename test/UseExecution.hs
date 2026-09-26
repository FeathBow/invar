{-# LANGUAGE OverloadedStrings #-}

module UseExecution (useExecution) where

import Data.List (sort)
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Use.Execution qualified as E

useExecution :: Group
useExecution =
    Group
        "Declared execution arrangements"
        [ ("every arrangement runs each input exactly once in bounded groups", withTests 200 (property partition))
        , ("a repeat must change the request order or the batch partition", withTests 1 (property schedules))
        ]

partition :: PropertyT IO ()
partition = do
    count <- forAll (Gen.integral (Range.linear 0 40))
    size <- forAll (Gen.integral (Range.linear 1 12))
    selected <- forAll (Gen.element [E.Declared, E.Reversed, E.Rotated 3, E.Rotated 41])
    let groups = E.arrange (E.Arrangement selected size) [0 .. count - 1 :: Int]
    sort (concat groups) === [0 .. count - 1]
    assert (all (\group -> not (null group) && length group <= fromIntegral size) groups)

schedules :: PropertyT IO ()
schedules = do
    let paired = E.Arrangement E.Declared 8
    E.unchanged 16 paired [E.Arrangement E.Declared 8] === [0]
    E.unchanged 16 paired [E.Arrangement (E.Rotated 16) 8] === [0]
    E.unchanged 16 paired [E.Arrangement E.Reversed 16, E.Arrangement E.Declared 16] === []
    E.unchanged 16 paired [E.Arrangement E.Reversed 8, E.Arrangement E.Reversed 8] === [0, 1]
    E.unchanged 16 paired [E.Arrangement E.Reversed 8, E.Arrangement (E.Rotated 3) 5] === []
