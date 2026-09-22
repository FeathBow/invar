{-# LANGUAGE Safe #-}

module Invar.Spec.Score where

import Data.ByteString (ByteString)
import Data.Word (Word32)
import Invar.Infer qualified as Infer
import Invar.Policy.Description qualified as Policy
import Invar.Spec.Invocation qualified as Invocation
import Numeric.Natural (Natural)

data Source = Source
    { sourceRequest :: Infer.Request
    , sourceBinding :: Invocation.Binding
    , sourcePolicy :: Policy.Description
    , sourceLog :: String
    }
    deriving (Eq, Show)

-- The numerator is the finite measured probability of the generating source.
-- A zero target probability therefore gives positive infinity, never NaN.
data LogRatio = Finite Rational | PositiveInfinity
    deriving (Eq, Show)

data Snapshot = Snapshot {step :: Natural, massWords :: [Word32]}
    deriving (Eq, Show)

data Distribution = Distribution {vocabulary :: Natural, snapshots :: [Snapshot]}
    deriving (Eq, Show)

data Fact = Fact
    { source :: Source
    , target :: Source
    , probabilityWords :: [Word32]
    , logRatio :: LogRatio
    , observation :: ByteString
    }
    deriving (Eq, Show)
