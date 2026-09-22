{-# LANGUAGE GHC2021 #-}

-- Reject: [GHC-01928]
module UseScope where

import Invar.Use

forge :: ScopeId -> Scope
forge identity = Scope identity "workload" "records" []
