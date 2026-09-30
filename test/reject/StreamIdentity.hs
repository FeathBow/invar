{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE Safe #-}

-- Reject: [GHC-22385]
module StreamIdentity (relabelled) where

import Invar.Learn.Stream (Stream, identity)

relabelled :: Stream -> Stream -> Stream
relabelled original changed = changed {identity = identity original}
