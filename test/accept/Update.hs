{-# LANGUAGE GHC2021 #-}

module Update (execute) where

import Invar.Learn qualified as L
import Invar.Learn.Worker qualified as W
import Invar.Spec.Invocation qualified as V
import Invar.Transcript qualified as Transcript

execute :: W.Worker -> V.Binding -> L.Plan scope -> IO (Either W.Failure (W.Execution scope))
execute worker binding planned = case W.prepare binding planned of
    Left problem -> pure (Left problem)
    Right call -> W.run worker Transcript.standard call
