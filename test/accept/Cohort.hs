{-# LANGUAGE GHC2021 #-}

module Cohort (collect) where

import Control.Monad (unless, zipWithM)
import Invar.Cohort qualified as C
import Invar.Infer.Result qualified as R

collect :: C.Definition -> [R.Result] -> Either C.Error (Either C.Error [Rational])
collect definition reports = C.withCohort definition $ \cohort -> do
    unless (length reports == length (C.members cohort)) (Left C.IncompleteCohort)
    observations <- zipWithM C.record (C.members cohort) reports
    batch <- C.admit cohort observations
    pure (map C.reward (C.observations batch))
