{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE DataKinds #-}

-- Reject: [GHC-18872]
module Scope (erase) where

import Data.Coerce (coerce)
import Invar.Construct qualified as C

erase :: C.Scoped '[ '( '[], Rational)] '[] Rational -> C.Flow '[] Rational
erase = coerce
