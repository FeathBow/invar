{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Invar.Construct (
    Origin (..),
    Flow,
    Scoped,
    Ref (..),
    Key,
    Record,
    Fields,
    Statement,
    Union,
    BuildError (..),
    literal,
    true,
    false,
    number,
    bits,
    token,
    source,
    booleanSource,
    numberSource,
    add,
    multiply,
    negateNumber,
    conjunction,
    negation,
    choose,
    variable,
    keyEqual,
    letValue,
    sequenceSource,
    mapSource,
    foldSequence,
    foldMap,
    mapValues,
    emptyFields,
    field,
    record,
    project,
    equal,
    emit,
    compile,
) where

import Control.Monad (unless)
import Data.Map.Strict qualified as Map
import Data.Proxy (Proxy (..))
import Data.Set (Set)
import Data.Word (Word32)
import GHC.TypeLits (KnownNat, KnownSymbol, Nat, symbolVal, type (<=))
import Invar.Construct.Types
import Invar.Literal qualified as L
import Invar.Spec.Artifact qualified as Artifact
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Operator qualified as O
import Invar.Spec.Program qualified as R
import Numeric.Natural (Natural)
import Prelude hiding (foldMap)

data Statement = Statement R.Command R.Type (Set R.Source, [Claim])

data BuildError
    = MissingSink String
    | AllowedMismatch String (Set R.Source) (Set R.Source)
    | InputMismatch String R.Type R.Type
    | MeaningMismatch String O.Operator O.Operator
    | MissingMeaning String
    | MissingSource R.Source
    | SourceMismatch R.Source R.Type R.Type
    | InvalidArtifact Artifact.LoadError
    deriving (Eq, Show)

literal :: forall value scope. (KnownValue value, KeyFree value) => Literal value -> Scoped scope '[] value
literal value = Term [] (const (R.Constant (valueType (Proxy @value)) (literalValue value)))

true, false :: Scoped scope '[] Bool
true = literal L.true
false = literal L.false

number :: forall (n :: Nat) (d :: Nat) scope. (KnownNat n, KnownNat d, 1 <= d) => Scoped scope '[] Rational
number = literal (L.number @n @d)

bits :: forall (n :: Nat) scope. (KnownNat n, n <= MaxBits) => Scoped scope '[] Word32
bits = literal (L.bits @n)

token :: forall (n :: Nat) scope. (KnownNat n) => Scoped scope '[] Natural
token = literal (L.token @n)

source :: forall origin value scope. (KnownOrigin origin, KnownValue value) => Scoped scope '[origin] value
source = sourceTerm @origin

booleanSource :: forall source scope. (KnownOrigin source) => Scoped scope '[source] Bool
booleanSource = sourceTerm @source

numberSource :: forall source scope. (KnownOrigin source) => Scoped scope '[source] Rational
numberSource = sourceTerm @source

sequenceSource :: forall source value scope. (KnownOrigin source, KnownValue value) => Scoped scope '[source] [value]
sequenceSource = sourceTerm @source

mapSource :: forall source value scope. (KnownOrigin source, KnownValue value) => Scoped scope '[source] (Map.Map Natural value)
mapSource = sourceTerm @source

sourceTerm :: forall source scope value. (KnownOrigin source, KnownValue value) => Scoped scope '[source] value
sourceTerm = Term [SourceType declared (valueType (Proxy @value))] (const (readSource declared))
  where
    declared = origin (Proxy @source)

readSource :: R.Source -> R.Expr
readSource input@(R.LogicalRandom _) = R.Read (R.Random input)
readSource input = R.Read (R.Input input)

add, multiply :: Scoped scope first Rational -> Scoped scope second Rational -> Scoped scope (Union first second) Rational
add (Term firstClaims first) (Term secondClaims second) = Term (firstClaims ++ secondClaims) (\depth -> R.Primitive "add" [first depth, second depth])
multiply (Term firstClaims first) (Term secondClaims second) = Term (firstClaims ++ secondClaims) (\depth -> R.Primitive "multiply" [first depth, second depth])

negateNumber :: Scoped scope sources Rational -> Scoped scope sources Rational
negateNumber (Term claims value) = Term claims (\depth -> R.Primitive "negate" [value depth])

conjunction :: Scoped scope first Bool -> Scoped scope second Bool -> Scoped scope (Union first second) Bool
conjunction (Term firstClaims first) (Term secondClaims second) = Term (firstClaims ++ secondClaims) (\depth -> R.Primitive "and" [first depth, second depth])

negation :: Scoped scope sources Bool -> Scoped scope sources Bool
negation (Term claims value) = Term claims (\depth -> R.Primitive "not" [value depth])

choose :: Scoped scope condition Bool -> Scoped scope first value -> Scoped scope second value -> Scoped scope (Union condition (Union first second)) value
choose (Term conditionClaims condition) (Term firstClaims first) (Term secondClaims second) = Term (conditionClaims ++ firstClaims ++ secondClaims) (\depth -> R.If (condition depth) (first depth) (second depth))

variable :: (KnownValue value) => Ref scope sources value -> Scoped scope sources value
variable ref = Term [] (R.Variable . reference ref)

reference :: Ref scope sources value -> Int -> String
reference Here depth = binder (pred depth)
reference (There ref) depth = reference ref (pred depth)

binder :: Int -> String
binder depth = "v" ++ show depth

keyEqual :: Ref scope first Key -> Ref scope second Key -> Scoped scope (Union first second) Bool
keyEqual first second = Term [] (\depth -> R.KeyEqual (reference first depth) (reference second depth))

letValue :: Scoped scope initial value -> Scoped ('(initial, value) ': scope) result output -> Scoped scope result output
letValue (Term initialClaims initial) (Term bodyClaims body) = Term (initialClaims ++ bodyClaims) (\depth -> R.Let (binder depth) (initial depth) (body (succ depth)))

foldSequence :: Scoped scope input [value] -> Scoped scope initial result -> Scoped ('(Union input initial, result) ': '(input, value) ': scope) body result -> Scoped scope (Union input (Union initial body)) result
foldSequence (Term inputClaims input) (Term initialClaims initial) (Term bodyClaims body) = Term (inputClaims ++ initialClaims ++ bodyClaims) $ \depth ->
    let accumulator = succ depth
     in R.Collect (R.FoldSequence (R.SequenceFold (input depth) (binder depth) (binder accumulator) (body (succ accumulator)) (initial depth)))

foldMap :: Scoped scope input (Map.Map Natural value) -> Scoped scope initial result -> Scoped ('(Union input initial, result) ': '(input, value) ': '(input, Key) ': scope) body result -> Scoped scope (Union input (Union initial body)) result
foldMap (Term inputClaims input) (Term initialClaims initial) (Term bodyClaims body) = Term (inputClaims ++ initialClaims ++ bodyClaims) $ \depth ->
    let payload = succ depth
        accumulator = succ payload
        bound = R.MapBody (binder depth) (binder payload) (body (succ accumulator))
     in R.Collect (R.FoldMap (R.MapFold (input depth) bound (binder accumulator) (initial depth)))

mapValues :: Scoped scope input (Map.Map Natural value) -> Scoped ('(input, value) ': '(input, Key) ': scope) body result -> Scoped scope (Union input body) (Map.Map Natural result)
mapValues (Term inputClaims input) (Term bodyClaims body) = Term (inputClaims ++ bodyClaims) $ \depth ->
    let payload = succ depth
        bound = R.MapBody (binder depth) (binder payload) (body (succ payload))
     in R.Collect (R.MapValues (input depth) bound)

emptyFields :: Fields scope '[] '[]
emptyFields = Fields [] (const Map.empty)

field :: forall name scope first value rest fields. (KnownSymbol name, Absent name fields) => Scoped scope first value -> Fields scope rest fields -> Fields scope (Union first rest) ('(name, value) ': fields)
field (Term valueClaims value) (Fields fieldClaims fields) = Fields (valueClaims ++ fieldClaims) (\depth -> Map.insert (symbolVal (Proxy @name)) (value depth) (fields depth))

record :: Fields scope sources fields -> Scoped scope sources (Record fields)
record (Fields claims fields) = Term claims (R.Fields . fields)

project :: forall name scope sources fields. (KnownSymbol name, KnownValue (Lookup name fields)) => Scoped scope sources (Record fields) -> Scoped scope sources (Lookup name fields)
project (Term claims value) = Term claims (\depth -> R.Project (value depth) (symbolVal (Proxy @name)))

equal :: forall name scope first second value. (KnownSymbol name, KnownValue value, KeyFree value) => Scoped scope first value -> Scoped scope second value -> Scoped scope (Union first second) Bool
equal (Term firstClaims first) (Term secondClaims second) = Term (Operation name (O.Equal kind) : firstClaims ++ secondClaims) (\depth -> R.Primitive name [first depth, second depth])
  where
    name = symbolVal (Proxy @name)
    kind = valueType (Proxy @value)

emit :: forall name spec allowed sources value. (KnownSymbol name, KnownSymbol spec, KnownOrigins allowed, KnownValue value, Allows sources allowed) => Flow sources value -> Statement
emit (Term claims value) = Statement (R.Emit (symbolVal (Proxy @name)) (symbolVal (Proxy @spec)) (value 0)) (valueType (Proxy @value)) (origins (Proxy @allowed), claims)

compile :: E.Semantics -> [Statement] -> Either BuildError Artifact.Checked
compile meaning statements = do
    checkMeanings meaning
    mapM_ (checkClaims meaning) statements
    commands <- traverse (bind (E.schema meaning)) statements
    either (Left . InvalidArtifact) Right (Artifact.load (Artifact.encode meaning commands))

bind :: R.Schema -> Statement -> Either BuildError R.Command
bind schema (Statement command@(R.Emit name _ _) kind (claimed, _)) = case Map.lookup name (R.sinks schema) of
    Nothing -> Left (MissingSink name)
    Just declared
        | claimed /= R.allowed declared -> Left (AllowedMismatch name claimed (R.allowed declared))
        | kind /= R.inputType declared -> Left (InputMismatch name kind (R.inputType declared))
        | otherwise -> Right command

checkClaims :: E.Semantics -> Statement -> Either BuildError ()
checkClaims meaning (Statement _ _ (_, claims)) = mapM_ check claims
  where
    check (SourceType input expected) = case Map.lookup input (R.sources (E.schema meaning)) of
        Nothing -> Left (MissingSource input)
        Just actual -> unless (actual == expected) (Left (SourceMismatch input expected actual))
    check (Operation name expected) = case Map.lookup name (E.meanings meaning) of
        Nothing -> Left (MissingMeaning name)
        Just actual -> unless (actual == expected) (Left (MeaningMismatch name expected actual))

checkMeanings :: E.Semantics -> Either BuildError ()
checkMeanings meaning = mapM_ check operations
  where
    operations = [("add", O.Add), ("multiply", O.Multiply), ("negate", O.Negate), ("and", O.And), ("not", O.Not)]
    check (name, expected) = case Map.lookup name (E.meanings meaning) of
        Nothing -> Right ()
        Just actual -> unless (actual == expected) (Left (MeaningMismatch name expected actual))
