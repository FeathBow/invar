{-# LANGUAGE GHC2021 #-}

module InitialConstructor (forge) where

import Invar.History.Initial qualified as Initial

-- Reject: [GHC-01928]
forge :: Initial.Checked
forge = Initial.Checked
