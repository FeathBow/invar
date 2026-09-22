{-# LANGUAGE GHC2021 #-}

module ScorePlan (forge) where

import Invar.Score qualified as S

-- Reject: [GHC-01928]
forge :: S.Plan
forge = S.Plan
