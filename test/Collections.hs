{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

module Collections (collections) where

import Control.Monad (forM_)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Construct qualified as C
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Operator qualified as O
import Invar.Spec.Program (Schema (..), Sink (..), Source (..), Type (..))
import Invar.Spec.Value (Scalar (..), Value (..))
import Numeric.Natural (Natural)
import Properties (campaign)

type Allowed = '[ 'C.Semantic "samples", 'C.Semantic "rewards", 'C.Semantic "nested", 'C.Semantic "offset"]

collections :: Group
collections =
    Group
        "Scoped collection construction"
        [ ("nested folds join keys rather than positions", once joins)
        , ("map-valued sequence elements retain lexical scope", once nested)
        , ("map-values keeps keys and captures the surrounding value", once mapped)
        , ("unused bindings do not taint the result", once unused)
        , ("folds retain collection and body dependencies", once dependency)
        , ("unused source reads retain their claimed types", once sourceClaims)
        , ("typed joins agree with an independent keyed reference", campaign reference)
        ]
  where
    once = withTests 1 . property

allowedSources :: Set.Set Source
allowedSources = Set.fromList [Semantic "samples", Semantic "rewards", Semantic "nested", Semantic "offset"]

meaning :: Type -> E.Semantics
meaning output = E.Semantics schema operations
  where
    operations = Map.fromList [("add", O.Add), ("multiply", O.Multiply), ("negate", O.Negate)]
    schema =
        Schema
            { sources = Map.fromList [(Semantic "samples", MapType NumberType), (Semantic "rewards", MapType NumberType), (Semantic "nested", SequenceType (MapType NumberType)), (Semantic "offset", NumberType), (Operational "history", NumberType)]
            , primitives = fmap O.signature operations
            , sinks = Map.singleton "out" (Sink "collection" output allowedSources Set.empty)
            }

inputs :: E.World
inputs =
    Map.fromList
        [ (Semantic "samples", numbers [(2, 3), (9, 8)])
        , (Semantic "rewards", numbers [(2, 20), (7, 70)])
        , (Semantic "nested", Sequence [numbers [(4, 10)], numbers [(2, 3), (9, 8)]])
        , (Semantic "offset", Atom (Number 5))
        , (Operational "history", Atom (Number 99))
        ]

numbers :: [(Natural, Rational)] -> Value Natural
numbers = Mapping . fmap (Atom . Number) . Map.fromList

joinProgram :: C.Statement
joinProgram = C.emit @"out" @"collection" @Allowed expression
  where
    expression = C.foldMap (C.mapSource @('C.Semantic "samples") @Rational) (C.number @0 @1) inner
    inner = C.foldMap (C.mapSource @('C.Semantic "rewards") @Rational) (C.variable C.Here) body
    body = C.add (C.variable C.Here) (C.choose equal (C.multiply outerValue (C.variable (C.There C.Here))) (C.number @0 @1))
    outerValue = C.variable (C.There (C.There (C.There (C.There C.Here))))
    equal = C.keyEqual (C.There (C.There C.Here)) (C.There (C.There (C.There (C.There (C.There C.Here)))))

result :: Rational -> Either E.Error [E.Emission]
result value = Right [E.Emission "out" "collection" (Atom (Number value))]

joins :: PropertyT IO ()
joins = do
    checked <- evalEither (C.compile (meaning NumberType) [joinProgram])
    A.run checked inputs === result 60
    A.run checked (Map.insert (Semantic "rewards") (numbers [(3, 20), (7, 70)]) inputs) === result 0
    A.run checked (Map.insert (Semantic "samples") (numbers []) inputs) === result 0

nested :: PropertyT IO ()
nested = do
    let expression = C.foldSequence (C.sequenceSource @('C.Semantic "nested") @(Map Natural Rational)) (C.number @0 @1) body
        body = C.foldMap (C.variable (C.There C.Here)) (C.variable C.Here) (C.add (C.variable C.Here) (C.variable (C.There C.Here)))
    checked <- evalEither (C.compile (meaning NumberType) [C.emit @"out" @"collection" @Allowed expression])
    A.run checked inputs === result 21
    A.run checked (Map.insert (Semantic "nested") (Sequence []) inputs) === result 0

mapped :: PropertyT IO ()
mapped = do
    let expression = C.letValue (C.numberSource @('C.Semantic "offset")) body
        body = C.mapValues (C.mapSource @('C.Semantic "samples") @Rational) (C.add (C.variable C.Here) (C.variable (C.There (C.There C.Here))))
    checked <- evalEither (C.compile (meaning (MapType NumberType)) [C.emit @"out" @"collection" @Allowed expression])
    A.run checked inputs === Right [E.Emission "out" "collection" (numbers [(2, 8), (9, 13)])]

unused :: PropertyT IO ()
unused = do
    let expression = C.letValue (C.numberSource @('C.Operational "history")) (C.number @1 @1)
    checked <- evalEither (C.compile (meaning NumberType) [C.emit @"out" @"collection" @Allowed expression])
    A.run checked inputs === result 1
    A.run checked (Map.insert (Operational "history") (Atom (Number (-1))) inputs) === result 1

dependency :: PropertyT IO ()
dependency = do
    let expression = C.foldMap (C.mapSource @('C.Semantic "samples") @Rational) (C.number @0 @1) (C.add (C.variable C.Here) (C.numberSource @('C.Semantic "offset")))
    checked <- evalEither (C.compile (meaning NumberType) [C.emit @"out" @"collection" @Allowed expression])
    A.run checked inputs === result 10
    A.run checked (Map.insert (Semantic "offset") (Atom (Number 7)) inputs) === result 14

maxEntries :: Int
maxEntries = 5

maxMagnitude :: Integer
maxMagnitude = 10

sourceClaims :: PropertyT IO ()
sourceClaims = forM_ cases $ \(expected, statement) ->
    case C.compile (meaning NumberType) [statement] of
        Left actual -> actual === expected
        Right _ -> failure
  where
    cases =
        [ (C.SourceMismatch (Operational "history") BooleanType NumberType, C.emit @"out" @"collection" @Allowed (C.letValue (C.booleanSource @('C.Operational "history")) (C.number @1 @1)))
        , (C.SourceMismatch (Semantic "samples") (MapType BooleanType) (MapType NumberType), C.emit @"out" @"collection" @Allowed (C.foldMap (C.mapSource @('C.Semantic "samples") @Bool) (C.number @0 @1) (C.add (C.variable C.Here) (C.number @1 @1))))
        , (C.MissingSource (Semantic "missing"), C.emit @"out" @"collection" @Allowed (C.letValue (C.numberSource @('C.Semantic "missing")) (C.number @1 @1)))
        ]

genNumbers :: Gen (Map Natural Rational)
genNumbers = Map.fromList <$> Gen.list (Range.linear 0 maxEntries) entry
  where
    entry = (,) <$> Gen.integral (Range.constant 0 (fromIntegral maxEntries)) <*> (fromInteger <$> Gen.integral (Range.linear (-maxMagnitude) maxMagnitude))

reference :: PropertyT IO ()
reference = do
    samples <- forAll genNumbers
    rewards <- forAll genNumbers
    history <- fromInteger <$> forAll (Gen.integral (Range.linear (-maxMagnitude) maxMagnitude))
    let expected = sum [value * reward | (key, value) <- Map.toList samples, Just reward <- [Map.lookup key rewards]]
        world = Map.insert (Semantic "samples") (numbers (Map.toList samples)) (Map.insert (Semantic "rewards") (numbers (Map.toList rewards)) inputs)
    cover 10 "shared keys" (not (Map.null (Map.intersection samples rewards)))
    cover 10 "unmatched keys" (Map.keysSet samples /= Map.keysSet rewards)
    checked <- evalEither (C.compile (meaning NumberType) [joinProgram])
    A.run checked world === result expected
    A.run checked (Map.insert (Operational "history") (Atom (Number history)) world) === result expected
