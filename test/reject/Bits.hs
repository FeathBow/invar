{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

-- Reject: Cannot satisfy: 4294967296 <=
module Bits (program) where

import Invar.Construct qualified as C

program :: C.Statement
program = C.emit @"out" @"value" @'[] (C.bits @4294967296)
