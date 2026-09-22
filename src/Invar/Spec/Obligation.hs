{-# LANGUAGE Safe #-}

module Invar.Spec.Obligation (Obligation (..)) where

import Data.ByteString (ByteString)

-- Shared by checked observation rules. The Evidence module remains the public
-- owner of obligation admission and reexports this unchanged record type.
data Obligation = Obligation
    { predicate :: String
    , specification :: ByteString
    , observation :: String
    , domain :: ByteString
    , binding :: ByteString
    }
    deriving (Eq, Show)
