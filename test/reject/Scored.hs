{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE Safe #-}

-- Reject: GHC-22385
module Scored where

import Invar.Reward (Scored, value)

replaceReward :: Scored -> Scored
replaceReward scored = scored {value = 1}
