{-# LANGUAGE Safe #-}

module Invar.Spec.Obligation (Obligation (..)) where

import Data.ByteString (ByteString)

data Obligation = Obligation
    { predicate :: String
    , specification :: ByteString
    , observation :: String
    , domain :: ByteString
    , binding :: ByteString
    }
    deriving (Eq, Show)
