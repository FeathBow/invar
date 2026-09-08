{-# LANGUAGE GHC2021 #-}

module Execution (forge) where

import Invar.Infer.Result qualified as R
import Invar.Spec.Invocation qualified as V
import Invar.Worker qualified as W

-- Reject: [GHC-01928]
forge :: V.Completion -> R.Result -> W.Execution
forge = W.Execution
