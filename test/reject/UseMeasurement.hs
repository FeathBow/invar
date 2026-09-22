{-# LANGUAGE GHC2021 #-}

-- Reject: [GHC-01928]
module UseMeasurement where

import Invar.Use.Measurement

forge :: Rational -> Measurement
forge = Measurement
