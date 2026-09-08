{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE DataKinds #-}

-- Reject: [GHC-18872]
module Source (erase) where

import Data.Coerce (coerce)
import Invar.Construct qualified as C

erase :: C.Flow '[ 'C.Operational "history"] Bool -> C.Flow '[] Bool
erase = coerce
