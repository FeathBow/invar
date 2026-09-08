{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

-- Reject: Missing field:
module Missing (program) where

import Invar.Construct qualified as C

program :: C.Statement
program = C.emit @"out" @"value" @'[] (C.project @"absent" (C.record C.emptyFields))
