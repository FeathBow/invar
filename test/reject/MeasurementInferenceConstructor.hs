{-# LANGUAGE GHC2021 #-}

module MeasurementInferenceConstructor (forge) where

import Invar.Measurement.Inference qualified as Measurement

-- Reject: [GHC-01928]
forge :: Measurement.Report
forge = Measurement.Report
