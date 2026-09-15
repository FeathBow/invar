{-# LANGUAGE GHC2021 #-}

module ReplayUpdateConstructor (forge) where

import Invar.Replay.Update qualified as Update

-- Reject: [GHC-01928]
forge :: Update.Update
forge = Update.Update
