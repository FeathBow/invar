{-# LANGUAGE OverloadedStrings #-}

module Evaluation (evaluation, semantics, world, genWorld, genExpression) where

import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Spec.Dependency qualified as D
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Operator qualified as O
import Invar.Spec.Program
import Invar.Spec.Value (Scalar (..), Value (..), normalForm)
import Numeric.Natural (Natural)
import Properties (campaign)

evaluation :: Group
evaluation =
    Group
        "Expression evaluation"
        [ ("pure scalar operators and records", once scalars)
        , ("canonical map and sequence traversal", once traversal)
        , ("cross-source joins compare bound identities", once joins)
        , ("map-values retains original keys", once mapping)
        , ("legal renaming preserves joins and anchored map outputs", once renaming)
        , ("logical randomness is read from the declared world", once randomness)
        , ("input worlds match their declared domains", once inputs)
        , ("missing or mismatched primitive meanings fail explicitly", once meanings)
        , ("commands bind checked sink outputs", once commands)
        , ("dependency agreement preserves evaluated values", campaign noninterference)
        ]
  where
    once = withTests 1 . property

own, history, samples, rewards, nested, randoms :: Source
own = Semantic "value"
history = Operational "history"
samples = Semantic "samples"
rewards = Semantic "rewards"
nested = Semantic "nested"
randoms = LogicalRandom "sampling"

semantics :: E.Semantics
semantics = E.Semantics declared operations
  where
    operations = Map.fromList [("and", O.And), ("not", O.Not), ("add", O.Add), ("multiply", O.Multiply), ("negate", O.Negate), ("equal", O.Equal NumberType)]
    declared =
        Schema
            { sources =
                Map.fromList
                    [ (own, NumberType)
                    , (history, BooleanType)
                    , (samples, MapType NumberType)
                    , (rewards, MapType NumberType)
                    , (nested, SequenceType (MapType NumberType))
                    , (randoms, MapType (SequenceType BitsType))
                    ]
            , primitives = fmap O.signature operations
            , sinks =
                Map.singleton
                    "observe"
                    Sink
                        { specification = "number-reference"
                        , inputType = NumberType
                        , allowed = Set.fromList [own, samples, rewards, nested, randoms]
                        , recorded = Set.empty
                        }
            }

number :: Rational -> Value Natural
number = Atom . Number

literal :: Rational -> Expr
literal value = Constant NumberType (number value)

readInput :: Source -> Expr
readInput = Read . Input

numberMap :: [(Natural, Rational)] -> Value Natural
numberMap = Mapping . Map.fromList . map (fmap number)

world :: E.World
world =
    Map.fromList
        [ (own, number 7)
        , (history, Atom (Boolean False))
        , (samples, numberMap [(9, 8), (2, 3)])
        , (rewards, numberMap [(2, 20), (7, 70)])
        , (nested, Sequence [numberMap [(2, 3)], numberMap [(9, 8)]])
        , (randoms, Mapping (Map.singleton 2 (Sequence [Atom (Bits32 0x3f000000)])))
        ]

mapFold :: Source -> MapBody -> Expr
mapFold source binder =
    Collect
        ( FoldMap
            MapFold
                { mapInput = readInput source
                , mapScope = binder
                , mapAccumulator = "acc"
                , mapInitial = literal 0
                }
        )

join :: Expr
join = mapFold samples (MapBody "sample" "value" inner)
  where
    inner =
        Collect
            ( FoldMap
                MapFold
                    { mapInput = readInput rewards
                    , mapScope = MapBody "reward" "score" body
                    , mapAccumulator = "subtotal"
                    , mapInitial = Variable "acc"
                    }
            )
    body = If (KeyEqual "sample" "reward") (Primitive "add" [Variable "subtotal", Variable "score"]) (Variable "subtotal")

sequenceFold :: Expr
sequenceFold =
    Collect
        ( FoldSequence
            SequenceFold
                { sequenceInput = readInput nested
                , itemName = "item"
                , sequenceAccumulator = "outer"
                , sequenceInitial = literal 0
                , sequenceExpression = inner
                }
        )
  where
    inner =
        Collect
            ( FoldMap
                MapFold
                    { mapInput = Variable "item"
                    , mapScope = MapBody "key" "value" (Variable "value")
                    , mapAccumulator = "inner"
                    , mapInitial = Variable "outer"
                    }
            )

scalars :: PropertyT IO ()
scalars = do
    let productExpr = Primitive "multiply" [literal 3, Primitive "negate" [literal 2]]
        fields = Fields (Map.fromList [("product", productExpr), ("own", readInput own)])
    E.evaluate semantics world (Project fields "product") === Right (number (-6))
    E.evaluate semantics world (Primitive "equal" [literal 3, Primitive "add" [literal 1, literal 2]]) === Right (Atom (Boolean True))
    E.evaluate semantics world (Primitive "not" [Primitive "and" [Read (Input history), Constant BooleanType (Atom (Boolean True))]]) === Right (Atom (Boolean True))
    E.evaluate semantics world (Let "bound" (readInput own) (Primitive "add" [Variable "bound", literal 2])) === Right (number 9)

traversal :: PropertyT IO ()
traversal = do
    let lastItem = mapFold samples (MapBody "key" "value" (Variable "value"))
    E.evaluate semantics world lastItem === Right (number 8)
    E.evaluate semantics (Map.insert samples (numberMap []) world) lastItem === Right (number 0)
    E.evaluate semantics world sequenceFold === Right (number 8)
    E.evaluate semantics (Map.insert nested (Sequence []) world) sequenceFold === Right (number 0)

joins :: PropertyT IO ()
joins = do
    E.evaluate semantics world join === Right (number 20)
    E.evaluate semantics (Map.insert rewards (numberMap [(3, 20), (7, 70)]) world) join === Right (number 0)

mapping :: PropertyT IO ()
mapping = do
    let expression = Collect (MapValues (readInput samples) (MapBody "key" "value" (Primitive "add" [Variable "value", readInput own])))
    E.evaluate semantics world expression === Right (numberMap [(2, 10), (9, 15)])

renaming :: PropertyT IO ()
renaming = do
    let base = Map.insert nested (Sequence []) (Map.insert randoms (Mapping Map.empty) world)
        first = Map.insert samples (numberMap [(1, 3), (2, 8)]) (Map.insert rewards (numberMap [(3, 20)]) base)
        second = Map.insert samples (numberMap [(2, 3), (3, 8)]) (Map.insert rewards (numberMap [(1, 20)]) base)
        visible assignment = Sequence (Map.elems (Map.delete history assignment))
        lastItem = mapFold samples (MapBody "key" "value" (Variable "value"))
        transformed = Collect (MapValues (readInput samples) (MapBody "key" "value" (Primitive "add" [Variable "value", readInput own])))
    normalForm (visible first) === normalForm (visible second)
    E.evaluate semantics first join === Right (number 0)
    E.evaluate semantics second join === Right (number 0)
    E.evaluate semantics first lastItem === Right (number 8)
    E.evaluate semantics second lastItem === Right (number 8)
    left <- evalEither (E.evaluate semantics first transformed)
    right <- evalEither (E.evaluate semantics second transformed)
    assert (left /= right)
    normalForm (Sequence [visible first, left]) === normalForm (Sequence [visible second, right])
    let broken = Map.insert samples (numberMap [(3, 3), (2, 8)]) second
    E.evaluate semantics broken lastItem === Right (number 3)
    assert (normalForm (visible first) /= normalForm (visible broken))

randomness :: PropertyT IO ()
randomness = E.evaluate semantics world (Read (Random randoms)) === Right (Mapping (Map.singleton 2 (Sequence [Atom (Bits32 0x3f000000)])))

inputs :: PropertyT IO ()
inputs = do
    E.evaluate semantics (Map.delete own world) (literal 0) === Left (E.MissingInput own)
    E.evaluate semantics (Map.insert own (Atom (Boolean False)) world) (readInput own) === Left (E.InvalidInput own NumberType)
    let unknown = Semantic "unknown"
    E.evaluate semantics (Map.insert unknown (number 0) world) (literal 0) === Left (E.ExtraInputs (Set.singleton unknown))

meanings :: PropertyT IO ()
meanings = do
    let absent = semantics {E.meanings = Map.delete "add" (E.meanings semantics)}
        mismatched = semantics {E.meanings = Map.insert "add" O.Not (E.meanings semantics)}
        call = Primitive "add" [literal 1, literal 2]
    E.evaluate absent world call === Left (E.MissingMeaning "add")
    E.evaluate mismatched world call === Left (E.MeaningMismatch "add" (O.signature O.Add) (O.signature O.Not))
    E.evaluate absent world (Let "unused" call (literal 0)) === Left (E.MissingMeaning "add")
    E.evaluate absent world (If (Read (Input history)) call (literal 0)) === Right (number 0)

commands :: PropertyT IO ()
commands = do
    E.runCommands semantics world [Emit "observe" "number-reference" join, Emit "observe" "number-reference" (readInput own)]
        === Right [E.Emission "observe" "number-reference" (number 20), E.Emission "observe" "number-reference" (number 7)]
    E.runCommands semantics world [Emit "observe" "number-reference" (If (readInput history) (literal 1) (literal 0))]
        === Left (E.InvalidProgram (D.ForbiddenSources (Set.singleton history)))

maxMagnitude, maxWidth, lastKey :: Int
maxMagnitude = 10
maxWidth = 3
lastKey = 3

genNumber :: Gen (Value Natural)
genNumber = number . fromIntegral <$> Gen.int (Range.linear (-maxMagnitude) maxMagnitude)

genMap :: Gen (Value Natural)
genMap = Mapping . Map.fromList <$> Gen.list (Range.linear 0 maxWidth) entry
  where
    entry = (,) <$> Gen.integral (Range.constant 0 (fromIntegral lastKey)) <*> genNumber

genWorld :: Gen E.World
genWorld = do
    ownValue <- genNumber
    historyValue <- Atom . Boolean <$> Gen.bool
    sampleValues <- genMap
    rewardValues <- genMap
    nestedValues <- Sequence <$> Gen.list (Range.linear 0 maxWidth) genMap
    pure (Map.fromList [(own, ownValue), (history, historyValue), (samples, sampleValues), (rewards, rewardValues), (nested, nestedValues)] <> Map.restrictKeys world (Set.singleton randoms))

genExpression :: Gen Expr
genExpression = Gen.recursive Gen.choice leaves branches
  where
    leaves = [literal . fromIntegral <$> Gen.int (Range.linear (-maxMagnitude) maxMagnitude), pure (readInput own), pure join, pure sequenceFold]
    branches =
        [ Primitive "add" <$> Gen.list (Range.singleton 2) genExpression
        , If (readInput history) <$> genExpression <*> genExpression
        , Let "unused" (readInput history) <$> genExpression
        , (\value -> Let "bound" value (Primitive "add" [Variable "bound", literal 1])) <$> genExpression
        ]

noninterference :: PropertyT IO ()
noninterference = do
    expression <- forAll genExpression
    first <- forAll genWorld
    candidate <- forAll genWorld
    analysis <- evalEither (D.analyze (E.schema semantics) expression)
    let second = Map.restrictKeys first (D.dependencies analysis) <> candidate
    cover 10 "operational input differs" (Map.lookup history first /= Map.lookup history second)
    cover 20 "outside-bound inputs differ" (first /= second)
    left <- evalEither (E.evaluate semantics first expression)
    right <- evalEither (E.evaluate semantics second expression)
    left === right
