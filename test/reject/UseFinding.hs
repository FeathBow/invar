{-# LANGUAGE GHC2021 #-}

module UseFinding where

import Invar.Spec.Evidence qualified as E
import Invar.Use qualified as U

-- Reject: [GHC-01928]
forge :: U.Claim -> E.Verdict -> U.Finding
forge = U.Finding
