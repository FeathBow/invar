{-# LANGUAGE OverloadedStrings #-}

module Objectives (objectives) where

import Control.Monad (forM_)
import Data.Either (isLeft)
import Data.Word (Word32)
import GHC.Float (castFloatToWord32)
import Hedgehog
import Invar.Learn.Objective qualified as O

objectives :: Group
objectives = Group "Core scalar objective" [("equal probability roles have the analytic loss and slope", once equalRoles), ("clipping and minimum ties retain the specified derivative", once boundaries), ("reference penalty has an independent analytic slope", once regularization), ("token mean follows logical order with one final division", once aggregation), ("signed zero and rounding are retained", once zeros), ("invalid inputs and intermediate overflow are rejected", once invalid)]
  where
    once = withTests 1 . property

word :: Float -> Word32
word = castFloatToWord32

inputs :: Float -> Float -> Float -> O.Inputs
inputs p old a = O.Inputs {O.current = word p, O.proximal = word old, O.behavior = word old, O.fixed = word p, O.advantage = word a}

equalRoles :: PropertyT IO ()
equalRoles = do
    actual <- evalEither (O.calculate (O.Profile 0.2 0.125) 2 [inputs (-1) (-1) 2, inputs (-1) (-1) (-2)])
    actual === [O.Output (word (-2)) (word (-1)) (word (-1)), O.Output (word 2) (word 1) (word 1)]
    O.mean32 (map O.term actual) === Right (word 0)

boundaries :: PropertyT IO ()
boundaries = do
    let profile = O.Profile 0.5 0
        -- These FP32 logarithms produce exactly the named ratios after exp.
        cases = [(-log 4, 0, 1, -0.25, -0.25), (-log 2, 0, 1, -0.5, -0.5), (0, 0, 1, -1, -1), (0, -log 1.5, 1, -1.5, -1.5), (0, -log 2, 1, -1.5, 0), (-log 4, 0, -1, 0.5, 0), (-log 2, 0, -1, 0.5, 0.5), (0, 0, -1, 1, 1), (0, -log 1.5, -1, 1.5, 1.5), (0, -log 2, -1, 2, 2)]
    forM_ cases $ \(p, old, a, value, slope) -> do
        [actual] <- evalEither (O.calculate profile 1 [inputs p old a])
        O.term actual === word value
        O.gradient actual === word slope

regularization :: PropertyT IO ()
regularization = do
    let logTwo = log 2
        source = (inputs (-logTwo) (-logTwo) 0) {O.fixed = word 0}
    [actual] <- evalEither (O.calculate (O.Profile 0.5 0.25) 2 [source])
    O.gradient actual === word (-0.125)
    O.rewardGradient actual === word (-0.0)
    O.term actual === word ((2 - logTwo - 1) * 0.25)

aggregation :: PropertyT IO ()
aggregation = do
    let large = 2 ** 24
        first = map word [large, 1, -large]
        second = map word [large, -large, 1]
    O.mean32 first === Right (word 0)
    O.mean32 second === Right (word (1 / 3))
    O.mean32 (map word [1, 2, 3]) === Right (word 2)
    O.mean32 (map word [1, 0, 0]) === Right (word (1 / 3))

zeros :: PropertyT IO ()
zeros = do
    [positive] <- evalEither (O.calculate (O.Profile 0.2 0) 1 [inputs 0 0 0])
    [negative] <- evalEither (O.calculate (O.Profile 0.2 0) 1 [inputs 0 0 (-0.0)])
    O.term positive === word 0
    O.term negative === word 0
    O.rewardGradient positive === word (-0.0)
    O.rewardGradient negative === word 0
    O.mean32 [word (-0.0)] === Right (word 0)
    let halfway = 2 ** (-24)
    O.mean32 (map word [1, halfway, -1]) === Right (word 0)

invalid :: PropertyT IO ()
invalid = do
    let profile = O.Profile 0.2 0.04
        ordinary = inputs (-1) (-1) 1
        nonfinite = [0x7f800000, 0xff800000, 0x7fc00000]
        bad = [ordinary {O.current = word 1}, ordinary {O.current = word (-1000)}, ordinary {O.proximal = word (-1000)}, ordinary {O.fixed = word (-1000)}, ordinary {O.current = word (-1e38), O.fixed = word 0}]
    forM_ bad $ \value -> assert (isLeft (O.calculate profile 1 [value]))
    forM_ nonfinite $ \value -> do
        assert (isLeft (O.calculate profile 1 [ordinary {O.advantage = value}]))
        assert (isLeft (O.mean32 [value]))
    forM_ [O.Profile 0 0, O.Profile 1 0, O.Profile (0 / 0) 0, O.Profile 0.2 (-1), O.Profile 0.2 (1 / 0), O.Profile 0.2 1e100] $ \value ->
        assert (isLeft (O.calculate value 1 [ordinary]))
    assert (isLeft (O.calculate profile 0 [ordinary]))
    assert (isLeft (O.calculate profile 1 []))
    assert (isLeft (O.calculate profile 1 [ordinary, ordinary]))
    assert (isLeft (O.mean32 []))
    assert (isLeft (O.mean32 (replicate 2 (word 3e38))))
