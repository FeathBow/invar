{-# LANGUAGE GHC2021 #-}

module HistoryGenerationConstructor (forge) where

import Invar.History.Generation qualified as Generation

-- Reject: [GHC-01928]
forge :: Generation.Generation
forge = Generation.Generation
