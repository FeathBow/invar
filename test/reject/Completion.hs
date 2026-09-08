{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE Safe #-}

-- Reject: GHC-22385
module Completion where

import Data.ByteString (ByteString)
import Invar.Spec.Invocation (Completion, completedOutput)

replaceOutput :: ByteString -> Completion -> Completion
replaceOutput output completed = completed {completedOutput = output}
