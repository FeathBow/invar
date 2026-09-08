{-# LANGUAGE GHC2021 #-}

module Observation (forge) where

import Invar.Cohort qualified as C
import Invar.Infer.Result qualified as R

-- Reject: [GHC-01928]
forge :: C.Member scope -> R.Result -> C.Observation scope
forge member result = C.Observation member result 1
