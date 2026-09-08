{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

module Literals (literals) where

import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Construct qualified as C
import Invar.Literal qualified as L
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Operator qualified as O
import Invar.Spec.Program (Schema (..), Sink (..), Source (..), Type (..))
import Invar.Spec.Value (Scalar (..), Value (..))
import Numeric.Natural (Natural)
import Properties (campaign)

literals :: Group
literals =
    Group
        "Closed key-free literals"
        [ ("nested sequences and records preserve literal payloads", once nested)
        , ("empty sequences retain their declared element types", once empty)
        , ("literal equality uses exact bit patterns", once equality)
        , ("generated literal sequences have no world dependencies", campaign reference)
        ]
  where
    once = withTests 1 . property

meaning :: Type -> E.Semantics
meaning payloadType = E.Semantics schema operations
  where
    operations = Map.singleton "equal-bits" (O.Equal (SequenceType BitsType))
    schema =
        Schema
            { sources = Map.singleton (Operational "history") BooleanType
            , primitives = fmap O.signature operations
            , sinks = Map.singleton "out" (Sink "literal" payloadType Set.empty Set.empty)
            }

world :: Bool -> E.World
world history = Map.singleton (Operational "history") (Atom (Boolean history))

output :: Value Natural -> Either E.Error [E.Emission]
output value = Right [E.Emission "out" "literal" value]

nested :: PropertyT IO ()
nested = do
    let fields = L.field @"values" (L.sequence [L.number @1 @2, L.negateNumber (L.number @7 @2)]) (L.field @"token" (L.token @99) L.emptyFields)
        value = L.sequence [L.record fields, L.record fields]
        kind = RecordType (Map.fromList [("values", SequenceType NumberType), ("token", TokenType)])
        expected = Record (Map.fromList [("values", Sequence [Atom (Number (1 / 2)), Atom (Number ((-7) / 2))]), ("token", Atom (Token 99))])
    checked <- evalEither (C.compile (meaning (SequenceType kind)) [C.emit @"out" @"literal" @'[] (C.literal value)])
    A.run checked (world False) === output (Sequence [expected, expected])

empty :: PropertyT IO ()
empty = do
    let value = L.sequence @(C.Record '[ '("tokens", [Natural])]) []
        kind = SequenceType (RecordType (Map.singleton "tokens" (SequenceType TokenType)))
    checked <- evalEither (C.compile (meaning kind) [C.emit @"out" @"literal" @'[] (C.literal value)])
    A.run checked (world False) === output (Sequence [])

equality :: PropertyT IO ()
equality = do
    let first = C.literal (L.sequence [L.bits @0, L.bits @2147483648])
        second = C.literal (L.sequence [L.bits @0, L.bits @0])
    checked <- evalEither (C.compile (meaning BooleanType) [C.emit @"out" @"literal" @'[] (C.equal @"equal-bits" first second)])
    A.run checked (world False) === output (Atom (Boolean False))

maxItems :: Int
maxItems = 8

reference :: PropertyT IO ()
reference = do
    flags <- forAll (Gen.list (Range.linear 0 maxItems) Gen.bool)
    let value = L.sequence [if flag then L.true else L.false | flag <- flags]
        expected = Sequence (map (Atom . Boolean) flags)
    cover 10 "empty sequence" (null flags)
    cover 20 "nonempty sequence" (not (null flags))
    checked <- evalEither (C.compile (meaning (SequenceType BooleanType)) [C.emit @"out" @"literal" @'[] (C.literal value)])
    A.run checked (world False) === output expected
    A.run checked (world True) === output expected
