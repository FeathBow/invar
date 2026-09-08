{-# LANGUAGE GHC2021 #-}

module CohortScope (mix) where

import Data.Kind (Type)
import Invar.Cohort qualified as C

-- Reject: [GHC-25897]
mix :: forall (first :: Type) (second :: Type). C.Cohort first -> [C.Observation second] -> Either C.Error (C.Batch first)
mix = C.admit
