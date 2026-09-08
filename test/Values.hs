{-# LANGUAGE OverloadedStrings #-}

module Values (values) where

import Data.List (permutations)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set (Set)
import Data.Set qualified as Set
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Spec.Value
import Properties (campaign)

type Sample = Value Int

values :: Group
values =
    Group
        "Joint keyed observations"
        [ ("joint observation preserves anchor identity", once anchor)
        , ("cross-map order is not observable", once crossOrder)
        , ("within-map order remains observable", once localOrder)
        , ("legal renaming depends on the visible structure", once composition)
        , ("scalar bits and structural shape remain observable", once structure)
        , ("normal equality characterizes legal renaming", campaign characterization)
        ]
  where
    once = withTests 1 . property

mapping :: [(Int, Bool)] -> Sample
mapping = Mapping . Map.fromList . map (fmap (Atom . Boolean))

anchor :: PropertyT IO ()
anchor = do
    let allowed = mapping [(1, False)]
        shared = mapping [(1, True)]
        separate = mapping [(2, True)]
    normalForm shared === normalForm separate
    assert (normalForm (Sequence [allowed, shared]) /= normalForm (Sequence [allowed, separate]))

crossOrder :: PropertyT IO ()
crossOrder = do
    let allowed = mapping [(2, False)]
        before = mapping [(1, True)]
        after = mapping [(3, True)]
    normalForm (Sequence [allowed, before]) === normalForm (Sequence [allowed, after])

localOrder :: PropertyT IO ()
localOrder = do
    let first = mapping [(1, False), (2, True)]
        shifted = mapping [(10, False), (20, True)]
        reversed = mapping [(10, True), (20, False)]
    normalForm first === normalForm shifted
    assert (normalForm first /= normalForm reversed)

composition :: PropertyT IO ()
composition = do
    let value = Sequence [mapping [(1, False), (2, True)], mapping [(3, False)]]
        rotation = Map.fromList [(1, 2), (2, 3), (3, 1)]
        once = rename rotation value
        twice = rename rotation once
    normalForm value === normalForm once
    assert (normalForm value /= normalForm twice)

structure :: PropertyT IO ()
structure = do
    let positiveZero = Atom (Bits32 0) :: Sample
        negativeZero = Atom (Bits32 0x80000000) :: Sample
    assert (normalForm positiveZero /= normalForm negativeZero)
    assert (normalForm (Sequence [] :: Sample) /= normalForm (Mapping (Map.empty :: Map Int Sample)))
    assert (normalForm (Record (Map.singleton "left" positiveZero)) /= normalForm (Record (Map.singleton "right" positiveZero)))

keyCount, maxWidth :: Int
keyCount = 3
maxWidth = 2

genValue :: Gen Sample
genValue = Gen.recursive Gen.choice atoms containers
  where
    atoms = [Atom . Boolean <$> Gen.bool, Atom . Number . fromIntegral <$> Gen.int (Range.linear 0 maxWidth)]
    children = Gen.list (Range.linear 0 maxWidth) genValue
    entry = (,) <$> Gen.int (Range.constant 0 (keyCount - 1)) <*> genValue
    containers =
        [ Sequence <$> children
        , Record . Map.fromList . zip ["left", "right"] <$> children
        , Mapping . Map.fromList <$> Gen.list (Range.linear 0 maxWidth) entry
        ]

characterization :: PropertyT IO ()
characterization = do
    first <- forAll genVisible
    related <- forAll Gen.bool
    order <- forAll (Gen.shuffle [0 .. keyCount - 1])
    second <- if related then pure (rename (Map.fromList (zip [0 ..] order)) first) else forAll genVisible
    let expected = relatedByRenaming first second
    cover 10 "key-carrying input" (not (Set.null (keys first)))
    cover 20 "shared key occurrences" (occurrences first > Set.size (keys first))
    cover 20 "renaming equivalent" expected
    cover 20 "distinct classes" (not expected)
    (normalForm first == normalForm second) === expected

genVisible :: Gen Sample
genVisible = Gen.choice [genValue, joint]
  where
    joint = do
        shared <- Gen.int (Range.constant 0 (keyCount - 1))
        other <- Gen.int (Range.constant 0 (keyCount - 1))
        first <- genValue
        second <- genValue
        let anchorMap = Mapping (Map.singleton shared first)
            joined = Mapping (Map.fromList [(shared, first), (other, second)])
        pure (Sequence [anchorMap, Record (Map.singleton "nested" joined)])

occurrences :: Sample -> Int
occurrences value = case value of
    Atom _ -> 0
    Record fields -> sum (fmap occurrences fields)
    Sequence items -> sum (map occurrences items)
    Mapping entries -> Map.size entries + sum (fmap occurrences entries)

rename :: Map Int Int -> Sample -> Sample
rename names value = case value of
    Atom scalar -> Atom scalar
    Record fields -> Record (fmap (rename names) fields)
    Sequence items -> Sequence (map (rename names) items)
    Mapping entries -> Mapping (Map.fromList (map entry (Map.toList entries)))
  where
    entry (key, payload) = case Map.lookup key names of
        Just changed -> (changed, rename names payload)
        Nothing -> error "Incomplete fixture renaming"

keys :: Sample -> Set Int
keys value = case value of
    Atom _ -> Set.empty
    Record fields -> foldMap keys fields
    Sequence items -> foldMap keys items
    Mapping entries -> Map.keysSet entries <> foldMap keys entries

relatedByRenaming :: Sample -> Sample -> Bool
relatedByRenaming first second
    | length domain /= length codomain = False
    | otherwise = any (\order -> under (Map.fromList (zip domain order)) first second) (permutations codomain)
  where
    domain = Set.toList (keys first)
    codomain = Set.toList (keys second)

under :: Map Int Int -> Sample -> Sample -> Bool
under names first second = case (first, second) of
    (Atom left, Atom right) -> left == right
    (Record left, Record right) -> paired field (Map.toList left) (Map.toList right)
    (Sequence left, Sequence right) -> paired (under names) left right
    (Mapping left, Mapping right) -> paired entry (Map.toAscList left) (Map.toAscList right)
    _ -> False
  where
    field (fieldName, left) (other, right) = fieldName == other && under names left right
    entry (key, left) (other, right) = Map.lookup key names == Just other && under names left right

paired :: (left -> right -> Bool) -> [left] -> [right] -> Bool
paired relation first second = length first == length second && and (zipWith relation first second)
