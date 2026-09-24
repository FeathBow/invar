{-# LANGUAGE Safe #-}

module Invar.Digest (sha256) where

sha256 :: String -> Bool
sha256 value = length value == 64 && all (`elem` (['0' .. '9'] ++ ['a' .. 'f'])) value
