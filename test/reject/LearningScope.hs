{-# LANGUAGE GHC2021 #-}

module LearningScope (mix) where

import Data.Coerce (coerce)
import Data.Kind (Type)
import Invar.Learn qualified as L

-- Reject: [GHC-25897]
mix :: forall (first :: Type) (second :: Type). L.Plan first -> L.Plan second
mix = coerce
