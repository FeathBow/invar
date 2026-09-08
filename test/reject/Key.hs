{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE Safe #-}

-- Reject: [GHC-39999]
module Key (expose) where

import Invar.Construct qualified as C

expose :: C.Ref scope sources C.Key -> C.Scoped scope sources C.Key
expose = C.variable
