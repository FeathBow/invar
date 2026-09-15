{-# LANGUAGE GHC2021 #-}

module HistoryCohortConstructor (forge) where

import Invar.History.Cohort qualified as Cohort

-- Reject: [GHC-01928]
forge :: Cohort.Checked
forge = Cohort.Checked
