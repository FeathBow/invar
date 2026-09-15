{-# LANGUAGE OverloadedStrings #-}

module Advantages (advantages) where

import Control.Monad (forM_)
import Data.Aeson qualified as Json
import Data.Map.Strict qualified as Map
import Data.Ratio ((%))
import Data.Word (Word32)
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Learn.Advantage qualified as A
import Invar.Learn.Program qualified as Program
import Invar.Learn.Wire qualified as Wire
import Invar.Spec.Artifact qualified as Artifact
import Invar.Spec.Program qualified as Source
import Invar.Spec.Value qualified as Value
import Learning (world)
import Properties (campaign)

advantages :: Group
advantages = Group "Core advantage reference" [("binary rewards have fixed FP32 words", once binary), ("equal decimal rewards follow rounded arithmetic", once rounded), ("exact summation retains cancellation terms", once summation), ("delivery permutations preserve keyed advantages", campaign permutations), ("grouping mutations change the actual advantage words", once grouping), ("wire normalization precedes the reference", once canonical), ("invalid groups and numerical failures are explicit", once invalid)]
  where
    once = withTests 1 . property

rewards :: [Double] -> [A.Reward]
rewards = zipWith (\index value -> A.Reward {A.sample = "s" ++ show (index :: Int), A.group = "g0", A.value = value}) [0 ..]

wordsFor :: [Double] -> PropertyT IO [Word32]
wordsFor supplied = Map.elems <$> evalEither (A.calculate delta (rewards supplied))

delta :: Double
delta = 0.0001

binary :: PropertyT IO ()
binary = do
    wordsFor [0, 1] >>= (=== [0xbf7ff2e5, 0x3f7ff2e5])
    wordsFor [0, 0] >>= (=== [0, 0])
    wordsFor [1, 1] >>= (=== [0, 0])
    wordsFor [-0.0, 0] >>= (=== [0x80000000, 0])

rounded :: PropertyT IO ()
rounded = wordsFor [0.1, 0.1, 0.1] >>= (=== replicate 3 0xaa1c4000)

summation :: PropertyT IO ()
summation = do
    A.sum64 [1e16, 1, -1e16] === Right 1
    A.sum64 [1, 2 ** (-53), 2 ** (-54)] === Right (1 + 2 ** (-52))
    A.sum64 [1e308, 1e308, -1e308] === Left (A.NonFinite "summation intermediate")

permutations :: PropertyT IO ()
permutations = do
    values <- forAll (Gen.list (Range.linear 2 32) (Gen.double (Range.linearFrac (-100) 100)))
    let supplied = rewards values
    reordered <- forAll (Gen.shuffle supplied)
    A.calculate delta reordered === A.calculate delta supplied

grouping :: PropertyT IO ()
grouping = do
    let supplied = zipWith (\assigned item -> item {A.group = assigned}) ["first", "first", "second", "second"] (rewards [0, 0, 1, 1])
        changed = zipWith (\assigned item -> item {A.group = assigned}) ["first", "second", "first", "second"] supplied
    before <- evalEither (A.calculate delta supplied)
    after <- evalEither (A.calculate delta changed)
    assert (before /= after)
    Map.elems before === replicate 4 0
    Map.elems after === [0xbf7ff2e5, 0xbf7ff2e5, 0x3f7ff2e5, 0x3f7ff2e5]

canonical :: PropertyT IO ()
canonical = do
    let small = (-1) % (10 ^ (400 :: Int))
        supplied = Map.insert (Source.Semantic "rewards") (Value.Mapping (Map.fromList [(2, Value.Atom (Value.Number small)), (9, Value.Atom (Value.Number 0))])) world
    checked <- evalEither Program.checked
    commands <- evalEither (Artifact.run checked supplied)
    forM_ commands $ \command -> do
        lowered <- evalEither (Wire.lower command)
        fields <- decode @(Map.Map String Json.Value) lowered
        samples <- evalMaybe (Map.lookup "samples" fields) >>= decode @[Map.Map String Json.Value]
        forM_ samples $ \sample -> do
            Map.lookup "reward" sample === Just (Json.Number 0)
            Map.lookup "advantage_bits" sample === Just (Json.Number 0)
  where
    decode :: forall result. (Json.FromJSON result) => Json.Value -> PropertyT IO result
    decode value = case Json.fromJSON value of
        Json.Success result -> pure result
        Json.Error problem -> annotate problem >> failure

invalid :: PropertyT IO ()
invalid = do
    forM_ [[], rewards [1], rewards [0 / 0, 0], rewards [1 / 0, 0], rewards [1e308, 1e308], rewards [1e200, -1e200]] $ \supplied ->
        case A.calculate delta supplied of
            Left _ -> success
            Right result -> annotateShow result >> failure
    forM_ [0, -1, 1 / 0, 0 / 0] $ \value ->
        case A.calculate value (rewards [0, 1]) of
            Left _ -> success
            Right result -> annotateShow result >> failure
    A.calculate delta [A.Reward "same" "g0" 0, A.Reward "same" "g0" 1] === Left (A.InvalidInput "Rewards must have distinct sample identities")
