{-# LANGUAGE GHC2021 #-}

module LearnerReceiptScope (mix) where

import Data.Coerce (coerce)
import Data.Kind (Type)
import Invar.Learn.Worker.Resident qualified as Resident

-- Reject: [GHC-25897]
mix :: forall (first :: Type) (second :: Type). Resident.Receipt first -> Resident.Receipt second
mix = coerce
