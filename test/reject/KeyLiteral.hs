{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE Safe #-}

-- Reject: Not a key-free payload:
module KeyLiteral (value) where

import Data.Map.Strict (Map)
import Invar.Literal qualified as L
import Numeric.Natural (Natural)

value :: L.Literal [Map Natural Bool]
value = L.sequence []
