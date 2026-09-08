{-# LANGUAGE GHC2021 #-}

module UpdateScope (mix) where

import Data.Coerce (coerce)
import Data.Kind (Type)
import Invar.Learn.Worker qualified as W

-- Reject: [GHC-25897]
mix :: forall (first :: Type) (second :: Type). W.Execution first -> W.Execution second
mix = coerce
