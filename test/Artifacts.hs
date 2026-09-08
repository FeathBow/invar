{-# LANGUAGE OverloadedStrings #-}

module Artifacts (artifacts) where

import Control.Monad (forM_)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Evaluation qualified as Examples
import Hedgehog
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Dependency qualified as D
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Operator qualified as O
import Invar.Spec.Program
import Invar.Spec.Value (Scalar (..), Value (..))
import Numeric.Natural (Natural)
import Properties (campaign)

artifacts :: Group
artifacts =
    Group
        "Finite program loading"
        [ ("independent text executes the bound program", once independent)
        , ("all input bytes belong to the program", once framing)
        , ("duplicate declarations and malformed values are rejected", once malformed)
        , ("unused declarations and primitive meanings are checked", once declarations)
        , ("serialized input cannot bypass key opacity", once opacity)
        , ("primitive substitution changes the fixed artifact", once binding)
        , ("all value forms retain their interpretation", once literals)
        , ("key-carrying input expressions survive serialization", once collections)
        , ("serialized programs preserve validation and evaluation", campaign correspondence)
        ]
  where
    once = withTests 1 . property

example :: ByteString
example =
    Bytes.unlines
        [ "(program 1"
        , " (sources ((semantic \"x\") number))"
        , " (signatures (\"add\" ((number number) number)))"
        , " (meanings (\"add\" add))"
        , " (sinks (\"out\" (\"sum\" number (allowed (semantic \"x\")) (recorded))))"
        , " (commands (emit \"out\" \"sum\" (primitive \"add\" (input (semantic \"x\")) (literal number (number 2 1))))))"
        ]

number :: Rational -> Value Natural
number = Atom . Number

world :: E.World
world = Map.singleton (Semantic "x") (number 3)

independent :: PropertyT IO ()
independent = do
    checked <- evalEither (A.load example)
    A.bytes checked === example
    A.run checked world === Right [E.Emission "out" "sum" (number 5)]

replace :: ByteString -> ByteString -> ByteString -> ByteString
replace old new input = case Bytes.breakSubstring old input of
    (_, rest) | Bytes.null rest -> error "Fixture replacement target is absent"
    (before, rest) -> before <> new <> Bytes.drop (Bytes.length old) rest

syntaxFailure :: ByteString -> PropertyT IO ()
syntaxFailure input = case A.load input of
    Left (A.SyntaxError _) -> success
    Left problem -> annotateShow problem >> failure
    Right _ -> failure

reject :: A.LoadError -> ByteString -> PropertyT IO ()
reject expected input = case A.load input of
    Left actual -> actual === expected
    Right _ -> failure

framing :: PropertyT IO ()
framing = do
    let trimmed = Bytes.dropWhileEnd (== '\n') example
    forM_ [0 .. Bytes.length trimmed - 1] $ \size -> syntaxFailure (Bytes.take size trimmed)
    syntaxFailure (example <> " (commands)")
    syntaxFailure (example <> ")")
    syntaxFailure (replace "program 1" "program 2" example)
    syntaxFailure (Bytes.pack ['\255'])

malformed :: PropertyT IO ()
malformed = do
    syntaxFailure (replace "(sources ((semantic \"x\") number))" "(sources ((semantic \"x\") number) ((semantic \"x\") bool))" example)
    syntaxFailure (replace "(number 2 1)" "(number 2 0)" example)
    syntaxFailure (replace "(number 2 1)" "(number 2 -1)" example)
    syntaxFailure (replace "(literal number (number 2 1))" "(literal bits (bits 4294967296))" example)
    syntaxFailure (replace "(literal number (number 2 1))" "(literal token (token -1))" example)
    syntaxFailure (replace "(allowed (semantic \"x\"))" "(allowed (semantic \"x\") (semantic \"x\"))" example)
    syntaxFailure (replace "(\"add\" add)" "(\"add\" callback)" example)

declarations :: PropertyT IO ()
declarations = do
    let noCommands = replace "(commands (emit \"out\" \"sum\" (primitive \"add\" (input (semantic \"x\")) (literal number (number 2 1)))))" "(commands)" example
    reject (A.MissingMeaning "add") (replace "(meanings (\"add\" add))" "(meanings)" noCommands)
    reject (A.MeaningMismatch "add" (O.signature O.Add) (O.signature O.Not)) (replace "(\"add\" add)" "(\"add\" not)" noCommands)
    reject (A.ExtraMeanings (Set.singleton "extra")) (replace "(meanings (\"add\" add))" "(meanings (\"add\" add) (\"extra\" add))" noCommands)
    reject (A.ValidationError (D.UnrecordedSources (Set.singleton (Operational "x")))) (replace "semantic" "operational" (replace "(allowed (semantic" "(allowed (operational" noCommands))

opacity :: PropertyT IO ()
opacity = do
    let typ = MapType NumberType
        val = Mapping (Map.singleton 1 (number 2))
    reject (A.ValidationError (D.KeyCarryingLiteral typ)) (A.encode (singleSink typ) [Emit "out" "value" (Constant typ val)])
    let empty = SequenceType typ
    reject (A.ValidationError (D.KeyCarryingLiteral empty)) (A.encode (singleSink empty) [Emit "out" "value" (Constant empty (Sequence []))])
    let op = O.Equal typ
        meaning = (singleSink BooleanType) {E.schema = (E.schema (singleSink BooleanType)) {primitives = Map.singleton "eq" (O.signature op)}, E.meanings = Map.singleton "eq" op}
    reject (A.ValidationError (D.KeyCarryingPrimitive "eq")) (A.encode meaning [])

binding :: PropertyT IO ()
binding = do
    original <- evalEither (A.load example)
    modified <- evalEither (A.load (replace "(\"add\" add)" "(\"add\" multiply)" example))
    assert (A.bytes original /= A.bytes modified)
    A.run original world === Right [E.Emission "out" "sum" (number 5)]
    A.run modified world === Right [E.Emission "out" "sum" (number 6)]

literals :: PropertyT IO ()
literals = forM_ examples $ \(typ, val) -> do
    let meaning = singleSink typ
        program = [Emit "out" "value" (Constant typ val)]
    checked <- evalEither (A.load (A.encode meaning program))
    A.run checked Map.empty === Right [E.Emission "out" "value" val]
  where
    examples =
        [ (BooleanType, Atom (Boolean True))
        , (NumberType, number (-(3 / 7)))
        , (BitsType, Atom (Bits32 0x80000000))
        , (TokenType, Atom (Token 17))
        , (RecordType (Map.singleton "\x03bb\"" NumberType), Record (Map.singleton "\x03bb\"" (number 2)))
        , (SequenceType BooleanType, Sequence [Atom (Boolean False), Atom (Boolean True)])
        ]

singleSink :: Type -> E.Semantics
singleSink typ = E.Semantics (Schema Map.empty Map.empty (Map.singleton "out" (Sink "value" typ Set.empty Set.empty))) Map.empty

collections :: PropertyT IO ()
collections = do
    forM_ expressions $ \expr -> do
        analysis <- evalEither (D.analyze (E.schema Examples.semantics) expr)
        let meaning = Examples.semantics {E.schema = (E.schema Examples.semantics) {sinks = Map.singleton "out" (Sink "value" (D.valueType analysis) (D.dependencies analysis) Set.empty)}}
            program = [Emit "out" "value" expr]
        checked <- evalEither (A.load (A.encode meaning program))
        A.run checked Examples.world === E.runCommands meaning Examples.world program
  where
    expressions =
        [ Read (Random (LogicalRandom "sampling"))
        , Collect (MapValues (Read (Input (Semantic "samples"))) (MapBody "key" "value" (Variable "value")))
        , Project (Fields (Map.singleton "map" (Read (Input (Semantic "samples"))))) "map"
        ]

correspondence :: PropertyT IO ()
correspondence = do
    expr <- forAll Examples.genExpression
    assignments <- forAll Examples.genWorld
    let meaning = Examples.semantics
        program = [Emit "observe" "number-reference" expr]
        encoded = A.encode meaning program
    case D.checkCommands (E.schema meaning) program of
        Left problem -> do
            cover 10 "forbidden program" True
            cover 10 "accepted program" False
            reject (A.ValidationError problem) encoded
        Right _ -> do
            cover 10 "forbidden program" False
            cover 10 "accepted program" True
            checked <- evalEither (A.load encoded)
            A.bytes checked === encoded
            A.run checked assignments === E.runCommands meaning assignments program
