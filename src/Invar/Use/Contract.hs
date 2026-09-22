{-# LANGUAGE Safe #-}

module Invar.Use.Contract (UseContract (..), module Invar.Spec.UseContract) where

import Invar.Policy.Description qualified as Policy
import Invar.Spec.Domain (Domain)
import Invar.Spec.Measurement (Method)
import Invar.Spec.UseContract
import Numeric.Natural (Natural)

data UseContract = UseContract
    { purpose :: String
    , declaredDomain :: Domain
    , declaredMeasurement :: Maybe Method
    , referenceImplementation :: Policy.Description
    , candidateImplementation :: Policy.Description
    , maximumContext :: Natural
    , criterion :: Criterion
    , freezeProtocol :: String
    , isolationProtocol :: String
    , selectionProtocol :: String
    , reliance :: [Reliance]
    }
    deriving (Eq, Show)
