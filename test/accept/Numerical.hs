{-# LANGUAGE GHC2021 #-}

module Numerical (inspect) where

import Invar.Numerical qualified as N

inspect :: N.BoundRun -> N.Relation -> Either N.ObservationError N.Finding
inspect supplied relation = do
    observed <- N.observe supplied
    pure (N.establish (N.Claim (N.scope observed) relation) observed)
