{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE Safe #-}

-- Reject: [GHC-01928]
module StepCompletion (forged) where

import Invar.Async.Completion (Completion (..))

forged :: Completion
forged = Completion 0 "consumed" "before" "after"
