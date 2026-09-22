{-# LANGUAGE GHC2021 #-}

module NumericalObservation (forge) where

import Invar.Numerical qualified as N

-- Reject: [GHC-01928]
forge :: N.Scope -> N.Path -> N.Observed
forge = N.Observed
