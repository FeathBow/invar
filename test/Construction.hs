{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

module Construction (construction) where

import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Construct qualified as C
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Dependency qualified as D
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Operator qualified as O
import Invar.Spec.Program
import Invar.Spec.Value (Scalar (..), Value (..))
import Properties (campaign)

type Allowed = '[ 'C.Semantic "value", 'C.Semantic "flag"]

construction :: Group
construction =
    Group
        "Source-indexed construction"
        [ ("arithmetic and semantic control reach the finite loader", once arithmetic)
        , ("Boolean construction preserves its declared sink type", once boolean)
        , ("type-level sink declarations match the loaded contract", once contracts)
        , ("primitive meanings cannot redefine typed operators", once meanings)
        , ("source type claims match the loaded schema", once sourceTypes)
        , ("host-level selection changes the program artifact", once selection)
        , ("typed output follows the arithmetic reference", campaign reference)
        ]
  where
    once = withTests 1 . property

allowedSources :: Set.Set Source
allowedSources = Set.fromList [Semantic "value", Semantic "flag"]

meaning :: E.Semantics
meaning = E.Semantics schema operations
  where
    operations = Map.fromList [("add", O.Add), ("multiply", O.Multiply), ("negate", O.Negate), ("and", O.And), ("not", O.Not)]
    schema =
        Schema
            { sources = Map.fromList [(Semantic "value", NumberType), (Semantic "flag", BooleanType), (Operational "history", BooleanType)]
            , primitives = fmap O.signature operations
            , sinks = Map.singleton "out" (Sink "arithmetic" NumberType allowedSources Set.empty)
            }

program :: C.Statement
program = C.emit @"out" @"arithmetic" @Allowed expression
  where
    expression = C.choose (C.booleanSource @('C.Semantic "flag")) (C.add value (C.multiply (C.number @1 @2) (C.number @1 @1))) (C.negateNumber value)
    value = C.numberSource @('C.Semantic "value")

inputs :: Rational -> Bool -> Bool -> E.World
inputs value flag history =
    Map.fromList [(Semantic "value", Atom (Number value)), (Semantic "flag", Atom (Boolean flag)), (Operational "history", Atom (Boolean history))]

arithmetic :: PropertyT IO ()
arithmetic = do
    checked <- evalEither (C.compile meaning [program])
    A.run checked (inputs 4 True False) === Right [E.Emission "out" "arithmetic" (Atom (Number (9 / 2)))]
    A.run checked (inputs 4 False True) === Right [E.Emission "out" "arithmetic" (Atom (Number (-4)))]

boolean :: PropertyT IO ()
boolean = do
    let adjusted = meaning {E.schema = (E.schema meaning) {sinks = Map.singleton "out" (Sink "boolean" BooleanType allowedSources Set.empty)}}
        expression = C.negation (C.conjunction (C.booleanSource @('C.Semantic "flag")) C.true)
    checked <- evalEither (C.compile adjusted [C.emit @"out" @"boolean" @Allowed expression])
    A.run checked (inputs 0 True False) === Right [E.Emission "out" "boolean" (Atom (Boolean False))]

rejected :: C.BuildError -> Either C.BuildError A.Checked -> PropertyT IO ()
rejected expected result = case result of
    Left actual -> actual === expected
    Right _ -> failure

contracts :: PropertyT IO ()
contracts = do
    rejected (C.AllowedMismatch "out" Set.empty allowedSources) (C.compile meaning [C.emit @"out" @"arithmetic" @'[] (C.number @1 @1)])
    rejected (C.InputMismatch "out" BooleanType NumberType) (C.compile meaning [C.emit @"out" @"arithmetic" @Allowed C.true])
    rejected (C.InvalidArtifact (A.ValidationError (D.WrongSpecification "arithmetic" "other"))) (C.compile meaning [C.emit @"out" @"other" @Allowed (C.number @1 @1)])
    rejected (C.MissingSink "missing") (C.compile meaning [C.emit @"missing" @"arithmetic" @Allowed (C.number @1 @1)])

meanings :: PropertyT IO ()
meanings = do
    let replaced = meaning {E.meanings = Map.insert "add" O.Multiply (E.meanings meaning)}
    rejected (C.MeaningMismatch "add" O.Add O.Multiply) (C.compile replaced [program])

sourceTypes :: PropertyT IO ()
sourceTypes = do
    let wrongType = meaning {E.schema = (E.schema meaning) {sources = Map.insert (Semantic "value") BooleanType (sources (E.schema meaning))}}
        statement = C.emit @"out" @"arithmetic" @Allowed (C.numberSource @('C.Semantic "value"))
    rejected (C.SourceMismatch (Semantic "value") NumberType BooleanType) (C.compile wrongType [statement])

selection :: PropertyT IO ()
selection = do
    let selected operational = if operational then C.number @1 @1 else C.number @0 @1
        statement operational = C.emit @"out" @"arithmetic" @Allowed (selected operational)
    first <- evalEither (C.compile meaning [statement False])
    second <- evalEither (C.compile meaning [statement True])
    assert (A.bytes first /= A.bytes second)
    A.run first (inputs 0 False False) === Right [E.Emission "out" "arithmetic" (Atom (Number 0))]
    A.run second (inputs 0 False False) === Right [E.Emission "out" "arithmetic" (Atom (Number 1))]

maxMagnitude :: Integer
maxMagnitude = 10

reference :: PropertyT IO ()
reference = do
    value <- fromInteger <$> forAll (Gen.integral (Range.linear (-maxMagnitude) maxMagnitude))
    flag <- forAll Gen.bool
    history <- forAll Gen.bool
    checked <- evalEither (C.compile meaning [program])
    let expected = if flag then value + 1 / 2 else negate value
    A.run checked (inputs value flag history) === Right [E.Emission "out" "arithmetic" (Atom (Number expected))]
    A.run checked (inputs value flag history) === A.run checked (inputs value flag (not history))
