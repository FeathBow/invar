{-# LANGUAGE GHC2021 #-}

module ScoredNumerical (inspect) where

import Invar.Numerical qualified as N
import Invar.Score qualified as S

inspect :: N.Run -> N.Run -> S.Report -> Either N.ObservationError N.Finding
inspect reference candidate score = do
    observed <- N.observe (N.ScoredRun reference candidate [(N.Reference, score)])
    pure (N.establish (N.Claim (N.scope observed) (N.ScoredPathLogRatioWithin N.Reference 0)) observed)
