{-# LANGUAGE GHC2021 #-}

module UngradedReward (rewarded) where

import Invar.Cohort qualified as C
import Invar.Rollout qualified as R

-- Reject: [GHC-83865]
rewarded :: R.SampleOf C.Ungraded -> Rational
rewarded = R.reward
