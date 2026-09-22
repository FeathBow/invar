{-# LANGUAGE GHC2021 #-}

module UseAdmission where

import Invar.Use qualified as U

-- Reject: [GHC-01928]
forge :: U.UseContract -> U.Admission
forge = U.Admission
