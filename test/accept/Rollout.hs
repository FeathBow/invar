{-# LANGUAGE GHC2021 #-}

module Rollout (collect) where

import Invar.Rollout qualified as R

collect :: R.Options -> IO (Either R.Error [Rational])
collect options = R.withDriver $ \driver -> fmap (map R.reward . R.samples) <$> R.run driver options
