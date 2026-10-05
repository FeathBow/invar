module Invar.Infer.Trajectory.Internal (Trajectory (..), Constraint (..), Observed (..)) where

import Invar.Infer.Result qualified as R
import Invar.Policy qualified as Policy
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as L

data Constraint = Materialization | Described Policy.Description
    deriving (Eq, Show)

data Observed = Observed
    { requested :: String
    , consumed :: String
    , tokenizer :: String
    , base :: String
    , assembly :: String
    , model :: String
    , revision :: String
    , fact :: L.Fact
    }

data Trajectory = Trajectory
    { completion :: V.Completion
    , constraint :: Constraint
    , observed :: Observed
    , result :: R.Result
    }
