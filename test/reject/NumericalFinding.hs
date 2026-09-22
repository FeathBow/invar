{-# LANGUAGE GHC2021 #-}

module NumericalFinding (forge) where

import Invar.Numerical qualified as N

-- Reject: [GHC-01928]
forge :: N.Finding
forge = N.Finding undefined undefined
