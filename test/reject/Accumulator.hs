{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

-- Reject: [GHC-83865]
module Accumulator (program) where

import Invar.Construct qualified as C

program :: C.Statement
program = C.emit @"out" @"value" @'[ 'C.Semantic "items"] (C.foldSequence (C.sequenceSource @('C.Semantic "items") @Rational) C.true (C.number @0 @1))
