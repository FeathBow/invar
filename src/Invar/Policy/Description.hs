{-# LANGUAGE Safe #-}

module Invar.Policy.Description (Description (..), bindings) where

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
