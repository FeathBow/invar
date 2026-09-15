{-# LANGUAGE GHC2021 #-}

module ReportConstructor (forge) where

import Invar.Evaluation qualified as Evaluation

-- Reject: [GHC-01928]
forge :: Evaluation.Report
forge = Evaluation.Report "input" "log" (Evaluation.Run "policy" 0) Evaluation.Unbound []
