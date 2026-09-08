{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE Safe #-}

-- Reject: it is a hidden module in the package
module Hidden where

import Invar.Construct.Types ()
