{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE Safe #-}

module Invar.Spec.Operator (
    Operator (..),
    OperatorError (..),
    signature,
    apply,
) where

import Control.DeepSeq (NFData)
import GHC.Generics (Generic)
import Invar.Spec.Program (Signature (..), Type (..), keyFree, matches)
import Invar.Spec.Value (Scalar (..), Value (..))
import Numeric.Natural (Natural)

data Operator = And | Not | Add | Multiply | Negate | Equal Type
    deriving (Eq, Show, Generic, NFData)

data OperatorError = InvalidSignature Signature | InvalidArguments Signature
    deriving (Eq, Show)

signature :: Operator -> Signature
signature operator = case operator of
    And -> Signature [BooleanType, BooleanType] BooleanType
    Not -> Signature [BooleanType] BooleanType
    Add -> Signature [NumberType, NumberType] NumberType
    Multiply -> Signature [NumberType, NumberType] NumberType
    Negate -> Signature [NumberType] NumberType
    Equal kind -> Signature [kind, kind] BooleanType

apply :: Operator -> [Value Natural] -> Either OperatorError (Value Natural)
apply operator values
    | not (all keyFree (result declared : arguments declared)) = Left (InvalidSignature declared)
    | not (validArguments declared values) = Left (InvalidArguments declared)
    | otherwise = calculate operator values
  where
    declared = signature operator

validArguments :: Signature -> [Value Natural] -> Bool
validArguments declared values =
    length (arguments declared) == length values && and (zipWith matches (arguments declared) values)

calculate :: Operator -> [Value Natural] -> Either OperatorError (Value Natural)
calculate operator values = case (operator, values) of
    (And, [Atom (Boolean first), Atom (Boolean second)]) -> Right (Atom (Boolean (first && second)))
    (Not, [Atom (Boolean value)]) -> Right (Atom (Boolean (not value)))
    (Add, [Atom (Number first), Atom (Number second)]) -> Right (Atom (Number (first + second)))
    (Multiply, [Atom (Number first), Atom (Number second)]) -> Right (Atom (Number (first * second)))
    (Negate, [Atom (Number value)]) -> Right (Atom (Number (negate value)))
    (Equal _, [first, second]) -> Right (Atom (Boolean (first == second)))
    _ -> Left (InvalidArguments (signature operator))
