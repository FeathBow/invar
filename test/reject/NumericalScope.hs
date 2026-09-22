{-# LANGUAGE GHC2021 #-}

module NumericalScope (forge) where

import Invar.Numerical qualified as N

-- Reject: [GHC-01928]
forge :: N.Scope
forge = N.Scope undefined undefined undefined undefined
