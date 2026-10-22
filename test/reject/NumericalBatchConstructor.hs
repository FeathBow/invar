{-# LANGUAGE GHC2021 #-}

module NumericalBatchConstructor (forge) where

import Invar.Numerical qualified as Numerical

-- Reject: [GHC-01928]
forge :: Numerical.Batch
forge = Numerical.Batch
