{-# LANGUAGE GHC2021 #-}

module ReplayOutputConstructor (forge) where

import Invar.Replay.Update.Output qualified as Output

-- Reject: [GHC-01928]
forge :: Output.Report
forge = Output.Report
