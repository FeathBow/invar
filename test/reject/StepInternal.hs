{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE Safe #-}

-- Reject: [GHC-87110]
module StepInternal (forged) where

import Invar.Async.Completion.Internal (Completion (..))

forged :: Completion
forged = Completion 0 "consumed" "before" "after"
