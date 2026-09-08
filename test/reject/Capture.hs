{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

-- Reject: [GHC-39999]
module Capture (capture) where

import Invar.Construct qualified as C

capture :: Rational -> C.Flow '[ 'C.Semantic "x"] Rational
capture history = fmap (+ history) (C.numberSource @('C.Semantic "x"))
