module Plan where

import Invar.Infer (Plan (..))

-- Reject: Illegal term-level use of the type constructor
unchecked :: Plan
unchecked = Plan ["--prompt=unchecked"]
