{-# LANGUAGE Safe #-}

module Invar.Async.Completion.Internal (Completion (..)) where

import Invar.Spec.Invocation (Binding)
import Numeric.Natural (Natural)

data Completion = Completion {binding :: Binding, plan :: String, step :: Natural, consumed :: String, before :: String, after :: String}
    deriving (Eq, Ord, Show)
