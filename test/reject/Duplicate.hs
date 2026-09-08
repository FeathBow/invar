{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

-- Reject: Duplicate field:
module Duplicate (program) where

import Invar.Construct qualified as C

program :: C.Statement
program = C.emit @"out" @"value" @'[] (C.record (C.field @"x" C.true (C.field @"x" C.false C.emptyFields)))
