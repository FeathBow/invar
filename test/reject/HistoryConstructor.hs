{-# LANGUAGE GHC2021 #-}

module HistoryConstructor (forge) where

import Invar.History qualified as History

-- Reject: [GHC-01928]
forge :: History.Checked
forge = History.Checked
