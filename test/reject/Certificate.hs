{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE Safe #-}

-- Reject: GHC-22385
module Certificate where

import Invar.Spec.Evidence (Certificate, assumptions)

erase :: Certificate -> Certificate
erase certificate = certificate {assumptions = []}
