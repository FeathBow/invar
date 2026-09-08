{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

-- Reject: Forbidden source:
module Projection (program) where

import Invar.Construct qualified as C

program :: C.Statement
program = C.emit @"out" @"value" @'[] (C.project @"x" (C.record (C.field @"x" C.true (C.field @"history" (C.booleanSource @('C.Operational "history")) C.emptyFields))))
