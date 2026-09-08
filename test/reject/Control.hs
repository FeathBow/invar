{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

-- Reject: Forbidden source:
module Control (program) where

import Invar.Construct qualified as C

program :: C.Statement
program = C.emit @"out" @"value" @'[] (C.choose (C.booleanSource @('C.Operational "history")) C.true C.false)
