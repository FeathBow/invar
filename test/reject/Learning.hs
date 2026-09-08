{-# LANGUAGE GHC2021 #-}

module Learning (forge) where

import Data.ByteString (ByteString)
import Invar.Learn qualified as L

-- Reject: [GHC-01928]
forge :: ByteString -> L.Plan scope
forge = L.Plan
