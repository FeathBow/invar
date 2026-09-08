{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

-- Reject: [GHC-83865]
module Expression (value) where

import Invar.Construct qualified as C
import Invar.Literal qualified as L

value :: L.Literal [Bool]
value = L.sequence [C.booleanSource @('C.Operational "history")]
