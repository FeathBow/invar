{-# LANGUAGE GHC2021 #-}

-- Reject: GHC-18872
module Live where

import Data.Coerce (coerce)
import Invar.Spec.Load (Fact, Live)

authorize :: Fact -> Live
authorize = coerce
