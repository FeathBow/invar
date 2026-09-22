{-# LANGUAGE Safe #-}

module Invar.Spec.Encode (document, value) where

import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Ratio (denominator, numerator)
import Data.Set qualified as Set
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Operator qualified as O
import Invar.Spec.Program
import Invar.Spec.Syntax
import Invar.Spec.Value (Scalar (..), Value (..))
import Numeric.Natural (Natural)

document :: E.Semantics -> [Command] -> Term
document meaning program =
    tagged
        "program"
        [ Bare "1"
        , tagged "sources" (mapping source kind (sources schema))
        , tagged "signatures" (mapping Quoted signature (primitives schema))
        , tagged "meanings" (mapping Quoted operator (E.meanings meaning))
        , tagged "sinks" (mapping Quoted sink (sinks schema))
        , tagged "commands" (map command program)
        ]
  where
    schema = E.schema meaning

mapping :: (key -> Term) -> (value -> Term) -> Map key value -> [Term]
mapping key payload = map (\(name, item) -> List [key name, payload item]) . Map.toAscList

source :: Source -> Term
source (Semantic name) = tagged "semantic" [Quoted name]
source (Operational name) = tagged "operational" [Quoted name]
source (LogicalRandom name) = tagged "random" [Quoted name]

kind :: Type -> Term
kind typ = case typ of
    BooleanType -> Bare "bool"
    NumberType -> Bare "number"
    BitsType -> Bare "bits"
    TokenType -> Bare "token"
    RecordType fields -> tagged "record" (mapping Quoted kind fields)
    SequenceType item -> tagged "sequence" [kind item]
    MapType item -> tagged "map" [kind item]

signature :: Signature -> Term
signature declared = List [List (map kind (arguments declared)), kind (result declared)]

operator :: O.Operator -> Term
operator op = case op of
    O.And -> Bare "and"
    O.Not -> Bare "not"
    O.Add -> Bare "add"
    O.Multiply -> Bare "multiply"
    O.Negate -> Bare "negate"
    O.Equal typ -> tagged "equal" [kind typ]

sink :: Sink -> Term
sink declared =
    List
        [ Quoted (specification declared)
        , kind (inputType declared)
        , tagged "allowed" (map source (Set.toAscList (allowed declared)))
        , tagged "recorded" (map source (Set.toAscList (recorded declared)))
        ]

value :: Value Natural -> Term
value val = case val of
    Atom (Boolean flag) -> tagged "bool" [Bare (if flag then "true" else "false")]
    Atom (Number number) -> tagged "number" [Bare (show (numerator number)), Bare (show (denominator number))]
    Atom (Bits32 bits) -> tagged "bits" [Bare (show bits)]
    Atom (Token token) -> tagged "token" [Bare (show token)]
    Record fields -> tagged "record" (mapping Quoted value fields)
    Sequence items -> tagged "sequence" (map value items)
    Mapping entries -> tagged "map" (mapping (Bare . show) value entries)

expression :: Expr -> Term
expression expr = case expr of
    Variable name -> tagged "var" [Quoted name]
    Read src -> readSource src
    Constant typ val -> tagged "literal" [kind typ, value val]
    Fields fields -> tagged "fields" (mapping Quoted expression fields)
    Project item name -> tagged "project" [expression item, Quoted name]
    If condition first second -> tagged "if" (map expression [condition, first, second])
    Let name item rest -> tagged "let" [Quoted name, expression item, expression rest]
    Primitive name args -> tagged "primitive" (Quoted name : map expression args)
    Collect exprs -> collection exprs
    KeyEqual first second -> tagged "keq" [Quoted first, Quoted second]

readSource :: ReadSource -> Term
readSource (Input src) = tagged "input" [source src]
readSource (Random src) = tagged "random" [source src]

collection :: Collection -> Term
collection (MapValues input binder) = tagged "mapv" [expression input, mapBody binder]
collection (FoldMap fold) =
    tagged "foldmap" [expression (mapInput fold), mapBody (mapScope fold), Quoted (mapAccumulator fold), expression (mapInitial fold)]
collection (FoldSequence fold) =
    tagged "foldseq" [expression (sequenceInput fold), Quoted (itemName fold), Quoted (sequenceAccumulator fold), expression (sequenceExpression fold), expression (sequenceInitial fold)]

mapBody :: MapBody -> Term
mapBody binder = List [Quoted (keyName binder), Quoted (valueName binder), expression (mapExpression binder)]

command :: Command -> Term
command (Emit name spec expr) = tagged "emit" [Quoted name, Quoted spec, expression expr]
