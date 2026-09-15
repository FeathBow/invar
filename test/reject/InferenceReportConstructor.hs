{-# LANGUAGE GHC2021 #-}

module InferenceReportConstructor (forge) where

import Invar.Infer.Observation qualified as Observation

-- Reject: [GHC-01928]
forge :: Observation.Report
forge = Observation.Report
