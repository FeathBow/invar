{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE Safe #-}

-- Reject: [GHC-22385]
module StepUpdate (moved) where

import Invar.Async.Completion (Completion, after)

moved :: Completion -> Completion
moved closed = closed {after = "elsewhere"}
