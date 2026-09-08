{-# LANGUAGE GHC2021 #-}

module BatchScope (mix) where

import Data.Coerce (coerce)
import Data.Kind (Type)
import Invar.Cohort qualified as C

-- Reject: [GHC-25897]
mix :: forall (first :: Type) (second :: Type). C.Batch first -> C.Batch second
mix = coerce
