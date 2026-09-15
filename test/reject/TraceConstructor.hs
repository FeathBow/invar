{-# LANGUAGE GHC2021 #-}

module TraceConstructor (forge) where

import Invar.History.Trace qualified as Trace

-- Reject: [GHC-01928]
forge :: Trace.Checked
forge = Trace.Checked
