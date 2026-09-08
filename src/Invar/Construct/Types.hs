{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RoleAnnotations #-}
{-# LANGUAGE Safe #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Invar.Construct.Types (
    Origin (..),
    Union,
    Allows,
    KnownOrigin (..),
    KnownOrigins (..),
    KnownValue (..),
    Scoped (..),
    Flow,
    Ref (..),
    Key,
    Record,
    Fields (..),
    Absent,
    Lookup,
    KeyFree,
    Claim (..),
    Literal (..),
    LiteralFields (..),
    literalValue,
    MaxBits,
) where

import Data.Kind (Constraint, Type)
import Data.Map.Strict qualified as Map
import Data.Proxy (Proxy (..))
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Word (Word32)
import GHC.TypeLits (ErrorMessage (..), KnownSymbol, Symbol, TypeError, symbolVal)
import Invar.Spec.Operator qualified as O
import Invar.Spec.Program qualified as R
import Invar.Spec.Value qualified as V
import Numeric.Natural (Natural)

data Origin = Semantic Symbol | Operational Symbol | LogicalRandom Symbol

type family Insert (source :: Origin) (sources :: [Origin]) :: [Origin] where
    Insert source '[] = '[source]
    Insert source (source ': rest) = source ': rest
    Insert source (other ': rest) = other ': Insert source rest

type family Union (first :: [Origin]) (second :: [Origin]) :: [Origin] where
    Union '[] second = second
    Union (source ': rest) second = Insert source (Union rest second)

type family Member (source :: Origin) (sources :: [Origin]) :: Constraint where
    Member source '[] = TypeError ('Text "Forbidden source: " ':<>: 'ShowType source)
    Member source (source ': rest) = ()
    Member source (other ': rest) = Member source rest

type family Allows (sources :: [Origin]) (allowed :: [Origin]) :: Constraint where
    Allows '[] allowed = ()
    Allows (source ': rest) allowed = (Member source allowed, Allows rest allowed)

class KnownOrigin (source :: Origin) where
    origin :: Proxy source -> R.Source

instance (KnownSymbol name) => KnownOrigin ('Semantic name) where
    origin _ = R.Semantic (symbolVal (Proxy @name))

instance (KnownSymbol name) => KnownOrigin ('Operational name) where
    origin _ = R.Operational (symbolVal (Proxy @name))

instance (KnownSymbol name) => KnownOrigin ('LogicalRandom name) where
    origin _ = R.LogicalRandom (symbolVal (Proxy @name))

class KnownOrigins (sources :: [Origin]) where
    origins :: Proxy sources -> Set R.Source

instance KnownOrigins '[] where
    origins _ = Set.empty

instance (KnownOrigin source, KnownOrigins rest) => KnownOrigins (source ': rest) where
    origins _ = Set.insert (origin (Proxy @source)) (origins (Proxy @rest))

data Claim = SourceType R.Source R.Type | Operation String O.Operator

type MaxBits = 4294967295

data Literal value where
    BooleanLiteral :: Bool -> Literal Bool
    NumberLiteral :: Rational -> Literal Rational
    BitsLiteral :: Word32 -> Literal Word32
    TokenLiteral :: Natural -> Literal Natural
    SequenceLiteral :: [Literal value] -> Literal [value]
    RecordLiteral :: LiteralFields fields -> Literal (Record fields)

data LiteralFields (fields :: [(Symbol, Type)]) where
    EmptyLiteralFields :: LiteralFields '[]
    FieldLiteral :: (KnownSymbol name) => Proxy name -> Literal value -> LiteralFields fields -> LiteralFields ('(name, value) ': fields)

literalValue :: Literal value -> V.Value Natural
literalValue literal = case literal of
    BooleanLiteral value -> V.Atom (V.Boolean value)
    NumberLiteral value -> V.Atom (V.Number value)
    BitsLiteral value -> V.Atom (V.Bits32 value)
    TokenLiteral value -> V.Atom (V.Token value)
    SequenceLiteral values -> V.Sequence (map literalValue values)
    RecordLiteral fields -> V.Record (literalFields fields)

literalFields :: LiteralFields fields -> Map.Map String (V.Value Natural)
literalFields EmptyLiteralFields = Map.empty
literalFields (FieldLiteral name value rest) = Map.insert (symbolVal name) (literalValue value) (literalFields rest)

type role Scoped nominal nominal nominal
data Scoped (scope :: [([Origin], Type)]) (sources :: [Origin]) (value :: Type) = Term [Claim] (Int -> R.Expr)

type Flow = Scoped '[]

data Key

data Ref (scope :: [([Origin], Type)]) (sources :: [Origin]) value where
    Here :: Ref ('(sources, value) ': scope) sources value
    There :: Ref scope sources value -> Ref (binding ': scope) sources value

type role Record nominal
data Record (fields :: [(Symbol, Type)])

type role Fields nominal nominal nominal
data Fields (scope :: [([Origin], Type)]) (sources :: [Origin]) (fields :: [(Symbol, Type)]) = Fields [Claim] (Int -> Map.Map String R.Expr)

type family Absent (name :: Symbol) (fields :: [(Symbol, Type)]) :: Constraint where
    Absent name '[] = ()
    Absent name ('(name, value) ': rest) = TypeError ('Text "Duplicate field: " ':<>: 'ShowType name)
    Absent name ('(other, value) ': rest) = Absent name rest

type family Lookup (name :: Symbol) (fields :: [(Symbol, Type)]) :: Type where
    Lookup name '[] = TypeError ('Text "Missing field: " ':<>: 'ShowType name)
    Lookup name ('(name, value) ': rest) = value
    Lookup name ('(other, value) ': rest) = Lookup name rest

class KnownValue value where
    valueType :: Proxy value -> R.Type

instance KnownValue Bool where
    valueType _ = R.BooleanType

instance KnownValue Rational where
    valueType _ = R.NumberType

instance KnownValue Word32 where
    valueType _ = R.BitsType

instance KnownValue Natural where
    valueType _ = R.TokenType

instance (KnownValue value) => KnownValue [value] where
    valueType _ = R.SequenceType (valueType (Proxy @value))

instance (KnownValue value) => KnownValue (Map.Map Natural value) where
    valueType _ = R.MapType (valueType (Proxy @value))

instance (KnownFields fields) => KnownValue (Record fields) where
    valueType _ = R.RecordType (fieldTypes (Proxy @fields))

class KnownFields (fields :: [(Symbol, Type)]) where
    fieldTypes :: Proxy fields -> Map.Map String R.Type

instance KnownFields '[] where
    fieldTypes _ = Map.empty

instance (KnownSymbol name, KnownValue value, KnownFields rest, Absent name rest) => KnownFields ('(name, value) ': rest) where
    fieldTypes _ = Map.insert (symbolVal (Proxy @name)) (valueType (Proxy @value)) (fieldTypes (Proxy @rest))

type family KeyFree value :: Constraint where
    KeyFree Bool = ()
    KeyFree Rational = ()
    KeyFree Word32 = ()
    KeyFree Natural = ()
    KeyFree [value] = KeyFree value
    KeyFree (Record fields) = KeyFreeFields fields
    KeyFree value = TypeError ('Text "Not a key-free payload: " ':<>: 'ShowType value)

type family KeyFreeFields (fields :: [(Symbol, Type)]) :: Constraint where
    KeyFreeFields '[] = ()
    KeyFreeFields ('(name, value) ': rest) = (KeyFree value, KeyFreeFields rest)
