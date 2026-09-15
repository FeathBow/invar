{-# LANGUAGE GHC2021 #-}

module MeasurementConstructor (forge) where

import Invar.Measurement qualified as Measurement

-- Reject: [GHC-01928]
forge :: Measurement.Report
forge = Measurement.Report
