{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE Safe #-}

module Invar.Spec.Program (
    Type (..),
    Source (..),
    Signature (..),
    Sink (..),
    Schema (..),
    ReadSource (..),
    Expr (..),
    Collection (..),
    MapBody (..),
    MapFold (..),
    SequenceFold (..),
    Command (..),
    keyFree,
    matches,
) where

import Control.DeepSeq (NFData)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set (Set)
import GHC.Generics (Generic)
import Invar.Spec.Value (Scalar (..), Value (..))
import Numeric.Natural (Natural)

data Type
    = BooleanType
    | NumberType
    | BitsType
    | TokenType
    | RecordType (Map String Type)
    | SequenceType Type
    | MapType Type
    deriving (Eq, Show, Generic, NFData)

data Source = Semantic String | Operational String | LogicalRandom String
    deriving (Eq, Ord, Show, Generic, NFData)

data Signature = Signature {arguments :: [Type], result :: Type}
    deriving (Eq, Show, Generic, NFData)

data Sink = Sink
    { specification :: String
    , inputType :: Type
    , allowed :: Set Source
    , recorded :: Set Source
    }
    deriving (Eq, Show, Generic, NFData)

data Schema = Schema
    { sources :: Map Source Type
    , primitives :: Map String Signature
    , sinks :: Map String Sink
    }
    deriving (Eq, Show, Generic, NFData)

data ReadSource = Input Source | Random Source
    deriving (Eq, Show, Generic, NFData)

data Expr
    = Variable String
    | Read ReadSource
    | Constant Type (Value Natural)
    | Fields (Map String Expr)
    | Project Expr String
    | If Expr Expr Expr
    | Let String Expr Expr
    | Primitive String [Expr]
    | Collect Collection
    | KeyEqual String String
    deriving (Eq, Show, Generic, NFData)

data Collection
    = FoldMap MapFold
    | FoldSequence SequenceFold
    | MapValues Expr MapBody
    deriving (Eq, Show, Generic, NFData)

data MapBody = MapBody
    { keyName :: String
    , valueName :: String
    , mapExpression :: Expr
    }
    deriving (Eq, Show, Generic, NFData)

data MapFold = MapFold
    { mapInput :: Expr
    , mapScope :: MapBody
    , mapAccumulator :: String
    , mapInitial :: Expr
    }
    deriving (Eq, Show, Generic, NFData)

data SequenceFold = SequenceFold
    { sequenceInput :: Expr
    , itemName :: String
    , sequenceAccumulator :: String
    , sequenceExpression :: Expr
    , sequenceInitial :: Expr
    }
    deriving (Eq, Show, Generic, NFData)

data Command = Emit String String Expr
    deriving (Eq, Show, Generic, NFData)

keyFree :: Type -> Bool
keyFree kind = case kind of
    MapType _ -> False
    SequenceType item -> keyFree item
    RecordType fields -> all keyFree fields
    _ -> True

matches :: Type -> Value key -> Bool
matches kind value = case (kind, value) of
    (BooleanType, Atom (Boolean _)) -> True
    (NumberType, Atom (Number _)) -> True
    (BitsType, Atom (Bits32 _)) -> True
    (TokenType, Atom (Token _)) -> True
    (RecordType fields, Record values) ->
        Map.keysSet fields == Map.keysSet values
            && and (Map.intersectionWith matches fields values)
    (SequenceType item, Sequence values) -> all (matches item) values
    (MapType item, Mapping values) -> all (matches item) values
    _ -> False
