{-# LANGUAGE GHC2021 #-}

module Driver (mix) where

import Data.Kind (Type)
import Invar.Rollout qualified as R

-- Reject: [GHC-25897]
mix :: forall (first :: Type) (second :: Type). R.Driver first -> R.Options -> IO (Either R.Error (R.Batch second))
mix = R.run
