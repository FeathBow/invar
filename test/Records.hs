{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

module Records (records) where

import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Word (Word32)
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

type Packet = C.Record '[ '("token", Natural), '("bits", Word32), '("value", Rational)]
type Nested = C.Record '[ '("items", Map Natural Rational)]
type Allowed = '[ 'C.Semantic "count", 'C.Semantic "packet", 'C.Semantic "nested", 'C.LogicalRandom "draws"]

records :: Group
records =
    Group
        "Typed structured payloads"
        [ ("record construction and projection reach their declared fields", once fields)
        , ("all scalar representations retain their exact values", once scalars)
        , ("records may carry maps without exposing keys", once nested)
        , ("typed logical randomness preserves sequence bits", once randomness)
        , ("equality binds a key-free type and its exact meaning", once equality)
        , ("record equality follows independent payload comparison", campaign reference)
        ]
  where
    once = withTests 1 . property

packetType :: Type
packetType = RecordType (Map.fromList [("token", TokenType), ("bits", BitsType), ("value", NumberType)])

meaning :: Type -> E.Semantics
meaning payloadType = E.Semantics schema operations
  where
    operations = Map.fromList [("add", O.Add), ("equal-packet", O.Equal packetType), ("equal-bool", O.Equal BooleanType)]
    schema =
        Schema
            { sources = Map.fromList [(Semantic "count", NumberType), (Semantic "packet", packetType), (Semantic "nested", RecordType (Map.singleton "items" (MapType NumberType))), (LogicalRandom "draws", SequenceType BitsType), (Operational "history", BooleanType)]
            , primitives = fmap O.signature operations
            , sinks = Map.singleton "out" (Sink "payload" payloadType (Set.fromList [Semantic "count", Semantic "packet", Semantic "nested", LogicalRandom "draws"]) Set.empty)
            }

packet :: Rational -> Word32 -> Value Natural
packet value bits = Record (Map.fromList [("token", Atom (Token 99)), ("bits", Atom (Bits32 bits)), ("value", Atom (Number value))])

inputs :: E.World
inputs =
    Map.fromList
        [ (Semantic "count", Atom (Number 4))
        , (Semantic "packet", packet 7 0)
        , (Semantic "nested", Record (Map.singleton "items" (Mapping (Map.fromList [(2, Atom (Number 3)), (9, Atom (Number 8))]))))
        , (LogicalRandom "draws", Sequence [Atom (Bits32 0), Atom (Bits32 0x80000000)])
        , (Operational "history", Atom (Boolean False))
        ]

constructed :: C.Flow '[ 'C.Semantic "count"] Packet
constructed = C.record (C.field @"token" (C.token @99) (C.field @"bits" (C.bits @0) (C.field @"value" value C.emptyFields)))
  where
    value = C.add (C.numberSource @('C.Semantic "count")) (C.number @1 @1)

output :: Value Natural -> Either E.Error [E.Emission]
output value = Right [E.Emission "out" "payload" value]

fields :: PropertyT IO ()
fields = do
    complete <- evalEither (C.compile (meaning packetType) [C.emit @"out" @"payload" @Allowed constructed])
    A.run complete inputs === output (packet 5 0)
    projected <- evalEither (C.compile (meaning NumberType) [C.emit @"out" @"payload" @Allowed (C.project @"value" (C.source @('C.Semantic "packet") @Packet))])
    A.run projected inputs === output (Atom (Number 7))

scalars :: PropertyT IO ()
scalars = do
    binary <- evalEither (C.compile (meaning BitsType) [C.emit @"out" @"payload" @Allowed (C.bits @4294967295)])
    A.run binary inputs === output (Atom (Bits32 0xffffffff))
    tokens <- evalEither (C.compile (meaning TokenType) [C.emit @"out" @"payload" @Allowed (C.token @18446744073709551616)])
    A.run tokens inputs === output (Atom (Token 18446744073709551616))
    signedZero <- evalEither (C.compile (meaning BitsType) [C.emit @"out" @"payload" @Allowed (C.project @"bits" (C.source @('C.Semantic "packet") @Packet))])
    A.run signedZero (Map.insert (Semantic "packet") (packet 7 0x80000000) inputs) === output (Atom (Bits32 0x80000000))

nested :: PropertyT IO ()
nested = do
    let expression = C.foldMap (C.project @"items" (C.source @('C.Semantic "nested") @Nested)) (C.number @0 @1) (C.add (C.variable C.Here) (C.variable (C.There C.Here)))
    checked <- evalEither (C.compile (meaning NumberType) [C.emit @"out" @"payload" @Allowed expression])
    A.run checked inputs === output (Atom (Number 11))

randomness :: PropertyT IO ()
randomness = do
    let expression = C.source @('C.LogicalRandom "draws") @[Word32]
    checked <- evalEither (C.compile (meaning (SequenceType BitsType)) [C.emit @"out" @"payload" @Allowed expression])
    A.run checked inputs === output (Sequence [Atom (Bits32 0), Atom (Bits32 0x80000000)])

equality :: PropertyT IO ()
equality = do
    let expression = C.equal @"equal-bool" C.false C.false
        statement = C.emit @"out" @"payload" @Allowed expression
        altered = (meaning BooleanType) {E.meanings = Map.insert "equal-bool" O.And (E.meanings (meaning BooleanType))}
    checked <- evalEither (C.compile (meaning BooleanType) [statement])
    A.run checked inputs === output (Atom (Boolean True))
    rejected (C.MeaningMismatch "equal-bool" (O.Equal BooleanType) O.And) (C.compile altered [statement])
    rejected (C.MissingMeaning "equal-bool") (C.compile altered {E.meanings = Map.delete "equal-bool" (E.meanings altered)} [statement])
    let unused = C.emit @"out" @"payload" @Allowed (C.letValue expression C.true)
    rejected (C.MeaningMismatch "equal-bool" (O.Equal BooleanType) O.And) (C.compile altered [unused])

rejected :: C.BuildError -> Either C.BuildError A.Checked -> PropertyT IO ()
rejected expected outcome = case outcome of
    Left actual -> actual === expected
    Right _ -> failure

maxMagnitude :: Integer
maxMagnitude = 10

reference :: PropertyT IO ()
reference = do
    count <- fromInteger <$> forAll (Gen.integral (Range.linear (-maxMagnitude) maxMagnitude))
    match <- forAll Gen.bool
    changeBits <- forAll Gen.bool
    history <- forAll Gen.bool
    let value = if match || changeBits then count + 1 else count
        bits = if not match && changeBits then 0x80000000 else 0
        world = Map.insert (Semantic "count") (Atom (Number count)) (Map.insert (Semantic "packet") (packet value bits) inputs)
        expression = C.equal @"equal-packet" constructed (C.source @('C.Semantic "packet") @Packet)
    cover 20 "equal payloads" match
    cover 20 "different payloads" (not match)
    checked <- evalEither (C.compile (meaning BooleanType) [C.emit @"out" @"payload" @Allowed expression])
    A.run checked world === output (Atom (Boolean match))
    A.run checked (Map.insert (Operational "history") (Atom (Boolean history)) world) === output (Atom (Boolean match))
