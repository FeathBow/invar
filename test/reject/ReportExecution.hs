{-# LANGUAGE GHC2021 #-}

module ReportExecution (forge) where

import Invar.Learn.Protocol qualified as Protocol
import Invar.Learn.Report qualified as Report

-- Reject: Couldn't match type
forge :: Report.Report -> Protocol.Result
forge = id
