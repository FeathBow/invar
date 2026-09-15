{-# LANGUAGE GHC2021 #-}

module CheckedConstructor (forge) where

import Invar.Spec.Artifact qualified as Artifact

-- Reject: [GHC-01928]
forge :: Artifact.Checked
forge = Artifact.Checked mempty (const (Right []))
