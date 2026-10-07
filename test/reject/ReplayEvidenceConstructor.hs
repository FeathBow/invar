{-# LANGUAGE GHC2021 #-}

module ReplayEvidenceConstructor (forge) where

import Invar.Async.Replay qualified as Replay

-- Reject: [GHC-01928]
forge :: Replay.Evidence
forge = Replay.Evidence
