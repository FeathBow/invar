{-# LANGUAGE GHC2021 #-}

module UseAssumption where

import Invar.Spec.Evidence qualified as E
import Invar.Use qualified as U

-- Reject: [GHC-83865]
assumed :: U.UseContract -> E.Verdict -> U.Decision
assumed = U.admit
