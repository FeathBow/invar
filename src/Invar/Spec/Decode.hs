{-# LANGUAGE Safe #-}

module Invar.Spec.Decode (document) where

import Control.Monad (foldM, unless)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Ratio ((%))
import Data.Set qualified as Set
import Data.Word (Word32)
import Invar.Spec.Operator qualified as O
import Invar.Spec.Program
import Invar.Spec.Syntax
import Invar.Spec.Value (Scalar (..), Value (..))
import Numeric.Natural (Natural)
import Text.Read (readMaybe)

document :: Term -> Either String (Schema, Map String O.Operator, [Command])
document (List [Bare "program", Bare "1", src, sig, ops, snk, cmds]) = do
    sourceTypes <- section "sources" src >>= mapping source kind
    signatures <- section "signatures" sig >>= mapping quoted signature
    operations <- section "meanings" ops >>= mapping quoted operator
    sinkTypes <- section "sinks" snk >>= mapping quoted sink
    program <- section "commands" cmds >>= traverse command
    pure (Schema sourceTypes signatures sinkTypes, operations, program)
document _ = Left "Expected program version 1 with sources, signatures, meanings, sinks and commands"

section :: String -> Term -> Either String [Term]
section expected (List (Bare actual : values)) | actual == expected = Right values
section expected _ = Left ("Expected " ++ expected ++ " section")

quoted :: Term -> Either String String
quoted (Quoted name) = Right name
quoted _ = Left "Expected a quoted name"

mapping :: (Ord key) => (Term -> Either String key) -> (Term -> Either String value) -> [Term] -> Either String (Map key value)
mapping key decodeValue = foldM insert Map.empty
  where
    insert entries (List [name, item]) = do
        parsed <- key name
        unless (Map.notMember parsed entries) (Left "Duplicate map entry")
        payload <- decodeValue item
        pure (Map.insert parsed payload entries)
    insert _ _ = Left "Expected a key-value pair"

source :: Term -> Either String Source
source (List [Bare tag, name]) = do
    label <- quoted name
    case tag of
        "semantic" -> Right (Semantic label)
        "operational" -> Right (Operational label)
        "random" -> Right (LogicalRandom label)
        _ -> Left "Unknown source class"
source _ = Left "Expected a source"

kind :: Term -> Either String Type
kind term = case term of
    Bare "bool" -> Right BooleanType
    Bare "number" -> Right NumberType
    Bare "bits" -> Right BitsType
    Bare "token" -> Right TokenType
    List [Bare "sequence", item] -> SequenceType <$> kind item
    List [Bare "map", item] -> MapType <$> kind item
    List (Bare "record" : fields) -> RecordType <$> mapping quoted kind fields
    _ -> Left "Unknown value type"

signature :: Term -> Either String Signature
signature (List [List args, output]) = Signature <$> traverse kind args <*> kind output
signature _ = Left "Expected a primitive signature"

operator :: Term -> Either String O.Operator
operator term = case term of
    Bare "and" -> Right O.And
    Bare "not" -> Right O.Not
    Bare "add" -> Right O.Add
    Bare "multiply" -> Right O.Multiply
    Bare "negate" -> Right O.Negate
    List [Bare "equal", item] -> O.Equal <$> kind item
    _ -> Left "Unknown primitive meaning"

sink :: Term -> Either String Sink
sink (List [spec, input, permitted, recordedInputs]) =
    Sink <$> quoted spec <*> kind input <*> sourceSet "allowed" permitted <*> sourceSet "recorded" recordedInputs
sink _ = Left "Expected a sink specification"

sourceSet :: String -> Term -> Either String (Set.Set Source)
sourceSet name term = do
    entries <- section name term >>= traverse source
    let unique = Set.fromList entries
    unless (Set.size unique == length entries) (Left "Duplicate source in sink declaration")
    pure unique

integer :: Term -> Either String Integer
integer (Bare text) = maybe (Left "Expected an integer") Right (readMaybe text)
integer _ = Left "Expected an integer"

natural :: Term -> Either String Natural
natural term = do
    parsed <- integer term
    if parsed < 0 then Left "Expected a nonnegative integer" else Right (fromInteger parsed)

value :: Term -> Either String (Value Natural)
value term = case term of
    List [Bare "bool", Bare "true"] -> Right (Atom (Boolean True))
    List [Bare "bool", Bare "false"] -> Right (Atom (Boolean False))
    List [Bare "number", n, d] -> rational n d
    List [Bare "bits", bits] -> bitPattern bits
    List [Bare "token", token] -> Atom . Token <$> natural token
    List (Bare "record" : fields) -> Record <$> mapping quoted value fields
    List (Bare "sequence" : items) -> Sequence <$> traverse value items
    List (Bare "map" : entries) -> Mapping <$> mapping natural value entries
    _ -> Left "Unknown value form"

rational :: Term -> Term -> Either String (Value Natural)
rational n d = do
    numerator <- integer n
    denominator <- integer d
    if denominator <= 0 then Left "Rational denominator must be positive" else Right (Atom (Number (numerator % denominator)))

bitPattern :: Term -> Either String (Value Natural)
bitPattern term = do
    bits <- natural term
    if bits > fromIntegral (maxBound :: Word32)
        then Left "Binary32 pattern exceeds 32 bits"
        else Right (Atom (Bits32 (fromIntegral bits)))

expression :: Term -> Either String Expr
expression term = case term of
    List [Bare "var", name] -> Variable <$> quoted name
    List [Bare "input", src] -> Read . Input <$> source src
    List [Bare "random", src] -> Read . Random <$> source src
    List [Bare "literal", typ, val] -> Constant <$> kind typ <*> value val
    List (Bare "fields" : fields) -> Fields <$> mapping quoted expression fields
    List [Bare "project", item, name] -> Project <$> expression item <*> quoted name
    List [Bare "if", condition, first, second] -> If <$> expression condition <*> expression first <*> expression second
    List [Bare "let", name, item, rest] -> Let <$> quoted name <*> expression item <*> expression rest
    _ -> operation term

operation :: Term -> Either String Expr
operation term = case term of
    List (Bare "primitive" : name : args) -> Primitive <$> quoted name <*> traverse expression args
    List [Bare "keq", first, second] -> KeyEqual <$> quoted first <*> quoted second
    List (Bare tag : parts) -> Collect <$> collection tag parts
    _ -> Left "Unknown expression form"

collection :: String -> [Term] -> Either String Collection
collection "mapv" [input, binder] = MapValues <$> expression input <*> mapBody binder
collection "foldmap" [input, binder, acc, initial] =
    FoldMap <$> (MapFold <$> expression input <*> mapBody binder <*> quoted acc <*> expression initial)
collection "foldseq" [input, item, acc, body, initial] =
    FoldSequence <$> (SequenceFold <$> expression input <*> quoted item <*> quoted acc <*> expression body <*> expression initial)
collection _ _ = Left "Unknown collection expression"

mapBody :: Term -> Either String MapBody
mapBody (List [key, item, body]) = MapBody <$> quoted key <*> quoted item <*> expression body
mapBody _ = Left "Expected map key, value and body bindings"

command :: Term -> Either String Command
command (List [Bare "emit", name, spec, body]) = Emit <$> quoted name <*> quoted spec <*> expression body
command _ = Left "Expected a top-level emission"
