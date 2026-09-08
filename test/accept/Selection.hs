{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

module Selection (select) where

import Invar.Construct qualified as C

select :: Bool -> C.Flow '[] Rational
select history = if history then C.number @1 @1 else C.number @0 @1
