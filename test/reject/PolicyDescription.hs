module PolicyDescription where

import qualified Invar.Policy as Policy

-- Reject: Data constructor out of scope
unchecked :: Policy.Description
unchecked = Policy.Description "" "" "" "" "" ""
