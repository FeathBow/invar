{-# LANGUAGE GHC2021 #-}

-- Reject: [GHC-01928]
module UseObservation where

import Invar.Use

forge :: Scope -> Observed
forge selected = Observed selected []
