{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

-- Reject: Not a key-free payload:
module KeyEquality (program) where

import Data.Map.Strict (Map)
import Invar.Construct qualified as C
import Numeric.Natural (Natural)

type Payload = C.Record '[ '("items", [Map Natural Bool])]

program :: C.Statement
program = C.emit @"out" @"value" @'[ 'C.Semantic "x"] (C.equal @"eq" (C.source @('C.Semantic "x") @Payload) (C.source @('C.Semantic "x") @Payload))
