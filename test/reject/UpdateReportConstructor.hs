{-# LANGUAGE GHC2021 #-}

module UpdateReportConstructor (forge) where

import Invar.Learn.Report qualified as Report

-- Reject: [GHC-01928]
forge :: Report.Report
forge = Report.Report
