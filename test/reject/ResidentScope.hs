{-# LANGUAGE GHC2021 #-}

module ResidentScope (mix) where

import Data.Coerce (coerce)
import Data.Kind (Type)
import Invar.Worker.Resident qualified as Resident

-- Reject: [GHC-25897]
mix :: forall (first :: Type) (second :: Type). Resident.Resident first -> Resident.Resident second
mix = coerce
