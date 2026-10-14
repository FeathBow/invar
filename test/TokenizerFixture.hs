{-# LANGUAGE OverloadedStrings #-}

module TokenizerFixture (description, describedAs, identity, coordinates) where

import Data.Aeson (Value, object, (.=))
import Invar.Tokenizer qualified as Tokenizer
import Numeric.Natural (Natural)

description :: Value
description = describedAs (replicate 64 '3') [(100000, "<eos>")]

describedAs :: String -> [(Natural, String)] -> Value
describedAs text added =
    object
        [ "format" .= ("invar-tokenizer-v2" :: String)
        , "token_space" .= object ["format" .= ("invar-token-space-v1" :: String), "vocabulary" .= object ["size" .= (100000 :: Natural), "sha256" .= replicate 64 '1'], "added" .= [object ["id" .= token, "content" .= content] | (token, content) <- added], "decoder" .= replicate 64 '2']
        , "text" .= text
        , "execution" .= replicate 64 '4'
        ]

identity :: String
identity = either error Tokenizer.identity (Tokenizer.describe description)

coordinates :: Value
coordinates = object ["input" .= (100352 :: Natural), "output" .= (100352 :: Natural)]
