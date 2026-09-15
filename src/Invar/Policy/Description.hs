{-# LANGUAGE Safe #-}

module Invar.Policy.Description (Description (..), bindings) where

-- The public policy reader validates these references before constructing a
-- description. Source provenance and numerical assembly remain separate fields.
data Description = Description
    { model :: String
    , revision :: String
    , adapter :: String
    , tokenizer :: String
    , base :: String
    , assembly :: String
    }
    deriving (Eq, Show)

bindings :: Description -> (String, String, String, String)
bindings selected = (adapter selected, tokenizer selected, base selected, assembly selected)
