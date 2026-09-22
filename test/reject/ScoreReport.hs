{-# LANGUAGE GHC2021 #-}

module ScoreReport (forge) where

import Invar.Score qualified as S

-- Reject: [GHC-01928]
forge :: S.Report
forge = S.Report
