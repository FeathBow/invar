{-# LANGUAGE GHC2021 #-}

module ReplayReferenceConstructor (forge) where

import Invar.Replay.Update qualified as Update

-- Reject: [GHC-01928]
forge :: Update.Reference
forge = Update.Reference
