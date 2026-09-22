{-# LANGUAGE GHC2021 #-}

module ScorePermit (forge) where

import Invar.Score qualified as S

-- Reject: [GHC-01928]
forge :: S.Permit
forge = S.Permit
