{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}
{-# LANGUAGE TypeFamilies #-}

module Invar.Literal (
    Literal,
    Fields,
    true,
    false,
    number,
    negateNumber,
    bits,
    token,
    sequence,
    emptyFields,
    field,
    record,
) where

import Data.Proxy (Proxy (..))
import Data.Ratio ((%))
import Data.Word (Word32)
import GHC.TypeLits (KnownNat, KnownSymbol, Nat, natVal, type (<=))
import Invar.Construct.Types (Absent, KeyFree, Literal (..), LiteralFields (..), MaxBits, Record)
import Numeric.Natural (Natural)
import Prelude hiding (sequence)

type Fields = LiteralFields

true, false :: Literal Bool
true = BooleanLiteral True
false = BooleanLiteral False

number :: forall (n :: Nat) (d :: Nat). (KnownNat n, KnownNat d, 1 <= d) => Literal Rational
number = NumberLiteral (natVal (Proxy @n) % natVal (Proxy @d))

negateNumber :: Literal Rational -> Literal Rational
negateNumber (NumberLiteral value) = NumberLiteral (negate value)

bits :: forall (n :: Nat). (KnownNat n, n <= MaxBits) => Literal Word32
bits = BitsLiteral (fromInteger (natVal (Proxy @n)))

token :: forall (n :: Nat). (KnownNat n) => Literal Natural
token = TokenLiteral (fromInteger (natVal (Proxy @n)))

sequence :: (KeyFree value) => [Literal value] -> Literal [value]
sequence = SequenceLiteral

emptyFields :: Fields '[]
emptyFields = EmptyLiteralFields

field :: forall name value fields. (KnownSymbol name, Absent name fields) => Literal value -> Fields fields -> Fields ('(name, value) ': fields)
field = FieldLiteral (Proxy @name)

record :: Fields fields -> Literal (Record fields)
record = RecordLiteral
