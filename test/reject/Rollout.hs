{-# LANGUAGE GHC2021 #-}

module Rollout (forge) where

import Invar.Rollout qualified as R

-- Reject: [GHC-01928]
forge :: [R.Sample] -> R.Batch scope
forge = R.Batch
