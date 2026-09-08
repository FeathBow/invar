{-# LANGUAGE GHC2021 #-}

module LoopScope (mix) where

import Data.Kind (Type)
import Invar.Loop qualified as L

-- Reject: [GHC-25897]
mix :: forall (first :: Type) (second :: Type). L.Driver first -> L.Cycle -> IO (Either L.Error (L.Generation second))
mix = L.run
