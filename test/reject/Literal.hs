{-# LANGUAGE GHC2021 #-}

-- Reject: [GHC-18872]
module Literal (change) where

import Data.Coerce (coerce)
import Invar.Literal qualified as L

change :: L.Literal Bool -> L.Literal Rational
change = coerce
