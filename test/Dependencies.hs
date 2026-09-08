{-# LANGUAGE OverloadedStrings #-}

module Dependencies (dependencies) where

import Control.Monad (forM_)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Hedgehog
import Invar.Spec.Dependency qualified as D
import Invar.Spec.Program
import Invar.Spec.Value (Scalar (..), Value (..))

dependencies :: Group
dependencies =
    Group
        "Program types and dependencies"
        [ ("input scope and literal types", once inputs)
        , ("control and unused bindings have distinct dependency rules", once control)
        , ("projection retains conservative record dependencies", once projection)
        , ("logical random sources are declared typed inputs", once randomness)
        , ("cross-traversal joins retain both sources", once joins)
        , ("key values cannot escape or survive lexical shadowing", once opacity)
        , ("map and sequence folds retain initial and body dependencies", once folds)
        , ("map construction preserves source provenance", once mapping)
        , ("primitive signatures exclude all key-carrying types", once signatures)
        , ("sinks check specification types and allowed dependencies", once emissions)
        ]
  where
    once = withTests 1 . property

low, high, samples, rewards, nested, randoms :: Source
low = Semantic "request"
high = Operational "arrival"
samples = Semantic "samples"
rewards = Semantic "rewards"
nested = Semantic "nested"
randoms = LogicalRandom "sampling"

schema :: Schema
schema =
    Schema
        { sources =
            Map.fromList
                [ (low, BooleanType)
                , (high, BooleanType)
                , (samples, MapType BooleanType)
                , (rewards, MapType BooleanType)
                , (nested, SequenceType (MapType BooleanType))
                , (randoms, MapType (SequenceType BitsType))
                ]
        , primitives = Map.fromList [("and", Signature [BooleanType, BooleanType] BooleanType)]
        , sinks = Map.fromList [(name, sink) | name <- ["decode", "score", "admit", "update", "publish"]]
        }
  where
    sink =
        Sink
            { specification = "boolean-reference"
            , inputType = BooleanType
            , allowed = Set.fromList [low, samples, rewards]
            , recorded = Set.empty
            }

boolean :: Bool -> Expr
boolean = Constant BooleanType . Atom . Boolean

readInput :: Source -> Expr
readInput = Read . Input

summary :: Type -> [Source] -> Either D.Error D.Analysis
summary kind = Right . D.Analysis kind . Set.fromList

inputs :: PropertyT IO ()
inputs = do
    D.analyze schema (boolean True) === summary BooleanType []
    D.analyze schema (readInput low) === summary BooleanType [low]
    D.analyze schema (readInput (Semantic "missing")) === Left (D.MissingSource (Semantic "missing"))
    D.analyze schema (Variable "missing") === Left (D.MissingVariable "missing")
    D.analyze schema (Constant BooleanType (Atom (Number 0))) === Left (D.InvalidLiteral BooleanType)
    D.analyze schema (Constant (SequenceType (MapType BooleanType)) (Sequence []))
        === Left (D.KeyCarryingLiteral (SequenceType (MapType BooleanType)))

control :: PropertyT IO ()
control = do
    D.analyze schema (If (readInput high) (boolean True) (boolean False)) === summary BooleanType [high]
    D.analyze schema (If (readInput high) (boolean False) (boolean False)) === summary BooleanType [high]
    D.analyze schema (Let "unused" (readInput high) (boolean False)) === summary BooleanType []
    D.analyze schema (Let "used" (readInput high) (Variable "used")) === summary BooleanType [high]
    D.analyze schema (If (readInput low) (boolean False) (readInput high)) === summary BooleanType [low, high]
    D.analyze schema (If (Constant NumberType (Atom (Number 0))) (boolean False) (boolean True))
        === Left (D.TypeMismatch BooleanType NumberType)
    D.analyze schema (Let "unused" (Variable "missing") (boolean False)) === Left (D.MissingVariable "missing")

projection :: PropertyT IO ()
projection = do
    let fields = Fields (Map.fromList [("allowed", readInput low), ("history", readInput high)])
    D.analyze schema (Project fields "allowed") === summary BooleanType [low, high]
    D.analyze schema (Project fields "missing") === Left (D.MissingField "missing")
    D.analyze schema (Project (boolean False) "missing") === Left (D.ExpectedRecord BooleanType)

randomness :: PropertyT IO ()
randomness = do
    D.analyze schema (Read (Random randoms)) === summary (MapType (SequenceType BitsType)) [randoms]
    D.analyze schema (Read (Random low)) === Left (D.NotRandom low)
    D.analyze schema (Read (Random (LogicalRandom "missing"))) === Left (D.MissingSource (LogicalRandom "missing"))

foldMapExpr :: Expr -> MapBody -> Expr -> Expr
foldMapExpr input binder seed =
    Collect
        ( FoldMap
            MapFold
                { mapInput = input
                , mapScope = binder
                , mapAccumulator = "acc"
                , mapInitial = seed
                }
        )

join :: Expr
join = foldMapExpr (readInput samples) (MapBody "sample" "value" inner) (boolean False)
  where
    inner = foldMapExpr (readInput rewards) (MapBody "reward" "score" (KeyEqual "sample" "reward")) (boolean False)

joins :: PropertyT IO ()
joins = do
    D.analyze schema join === summary BooleanType [samples, rewards]
    D.analyze schema (KeyEqual "sample" "sample") === Left (D.MissingVariable "sample")
    D.analyze schema (Let "sample" (boolean False) (KeyEqual "sample" "sample")) === Left (D.NotKey "sample")

opacity :: PropertyT IO ()
opacity = do
    let escape = MapBody "key" "value" (Variable "key")
        shadow = MapBody "key" "value" (Let "key" (boolean False) (KeyEqual "key" "key"))
        duplicate = MapBody "same" "same" (boolean False)
    D.analyze schema (foldMapExpr (readInput samples) escape (boolean False)) === Left (D.KeyEscapes "key")
    D.analyze schema (foldMapExpr (readInput samples) shadow (boolean False)) === Left (D.NotKey "key")
    D.analyze schema (foldMapExpr (readInput samples) duplicate (boolean False)) === Left (D.DuplicateBinder "same")

folds :: PropertyT IO ()
folds = do
    let body = MapBody "key" "value" (If (Variable "acc") (Variable "value") (readInput high))
        traversal =
            SequenceFold
                { sequenceInput = readInput nested
                , itemName = "item"
                , sequenceAccumulator = "outer"
                , sequenceExpression = foldMapExpr (Variable "item") (MapBody "key" "value" (Variable "outer")) (readInput low)
                , sequenceInitial = readInput high
                }
    D.analyze schema (foldMapExpr (readInput samples) body (readInput low)) === summary BooleanType [samples, low, high]
    D.analyze schema (Collect (FoldSequence traversal)) === summary BooleanType [nested, low, high]
    D.analyze schema (foldMapExpr (readInput samples) (MapBody "key" "value" (readInput rewards)) (boolean False))
        === Left (D.TypeMismatch BooleanType (MapType BooleanType))
    D.analyze schema (Collect (FoldSequence traversal {sequenceInput = readInput low})) === Left (D.ExpectedSequence BooleanType)

mapping :: PropertyT IO ()
mapping = do
    let body = MapBody "key" "value" (readInput low)
    D.analyze schema (Collect (MapValues (readInput samples) body)) === summary (MapType BooleanType) [samples, low]
    D.analyze schema (Collect (MapValues (readInput low) body)) === Left (D.ExpectedMap BooleanType)
    D.analyze schema (Collect (MapValues (readInput samples) (MapBody "key" "value" (readInput rewards))))
        === summary (MapType (MapType BooleanType)) [samples, rewards]

signatures :: PropertyT IO ()
signatures = do
    D.analyze schema (Primitive "and" [readInput low, readInput high]) === summary BooleanType [low, high]
    D.analyze schema (Primitive "and" []) === Left (D.ArgumentTypes [BooleanType, BooleanType] [])
    D.analyze schema (Primitive "missing" []) === Left (D.MissingPrimitive "missing")
    forM_ [MapType BooleanType, SequenceType (MapType BooleanType), RecordType (Map.singleton "map" (MapType BooleanType))] $ \kind -> do
        let forbidden = schema {primitives = Map.singleton "probe" (Signature [kind] BooleanType)}
            producing = schema {primitives = Map.singleton "probe" (Signature [] kind)}
        D.analyze forbidden (Primitive "probe" []) === Left (D.KeyCarryingPrimitive "probe")
        D.analyze producing (Primitive "probe" []) === Left (D.KeyCarryingPrimitive "probe")

emissions :: PropertyT IO ()
emissions = do
    forM_ ["decode", "score", "admit", "update", "publish"] $ \sink -> do
        D.checkCommands schema [Emit sink "boolean-reference" join] === sequence [summary BooleanType [samples, rewards]]
        D.checkCommands schema [Emit sink "boolean-reference" (If (readInput high) (boolean True) (boolean False))]
            === Left (D.ForbiddenSources (Set.singleton high))
        D.checkCommands schema [Emit sink "boolean-reference" (If (readInput high) (boolean False) (boolean False))]
            === Left (D.ForbiddenSources (Set.singleton high))
    D.checkCommands schema [Emit "decode" "other" (boolean False)] === Left (D.WrongSpecification "boolean-reference" "other")
    D.checkCommands schema [Emit "missing" "boolean-reference" (boolean False)] === Left (D.MissingSink "missing")
    let widened = schema {sinks = fmap (\sink -> sink {allowed = Set.insert high (allowed sink)}) (sinks schema)}
        recordedChoice = widened {sinks = fmap (\sink -> sink {recorded = Set.singleton high}) (sinks widened)}
    D.checkCommands widened [Emit "decode" "boolean-reference" (readInput high)] === Left (D.UnrecordedSources (Set.singleton high))
    D.checkCommands recordedChoice [Emit "decode" "boolean-reference" (readInput high)] === sequence [summary BooleanType [high]]
