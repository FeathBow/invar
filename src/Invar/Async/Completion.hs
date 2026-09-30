{-# LANGUAGE Safe #-}

module Invar.Async.Completion (Completion, binding, plan, step, consumed, before, after) where

import Invar.Async.Completion.Internal (Completion)
import Invar.Async.Completion.Internal qualified as Internal
import Invar.Spec.Invocation (Binding)
import Numeric.Natural (Natural)

binding :: Completion -> Binding
binding = Internal.binding

plan :: Completion -> String
plan = Internal.plan

step :: Completion -> Natural
step = Internal.step

consumed :: Completion -> String
consumed = Internal.consumed

before :: Completion -> String
before = Internal.before

after :: Completion -> String
after = Internal.after
