{-# LANGUAGE GHC2021 #-}

module LearnPermit (forge) where

import Invar.Learn.Protocol qualified as P

-- Reject: [GHC-01928]
forge :: P.Permit
forge = P.Permit
