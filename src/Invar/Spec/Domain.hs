{-# LANGUAGE Safe #-}

module Invar.Spec.Domain (Key (..), Input (..), Domain (..)) where

import Data.ByteString (ByteString)
import Data.List.NonEmpty (NonEmpty)
import Data.Map.Strict (Map)
import Invar.Spec.Value (Value)
import Numeric.Natural (Natural)

data Key = Key {cohort :: Natural, task :: String}
    deriving (Eq, Ord, Show)

-- An input declaration is known before execution. Parameter values have no
-- built-in task semantics; an identified measurement method binds their roles.
data Input = Input
    { inputKey :: Key
    , unitId :: String
    , prompt :: String
    , tokens :: Natural
    , temperature :: Double
    , seed :: Integer
    , parameters :: Map String (Value Natural)
    }
    deriving (Eq, Show)

data Domain = Domain
    { domainName :: String
    , provenance :: ByteString
    , unitDefinition :: String
    , declaredInputs :: NonEmpty Input
    }
    deriving (Eq, Show)
