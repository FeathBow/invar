{-# LANGUAGE OverloadedStrings #-}

module Tokenizers (tokenizers) where

import Data.Aeson (Value (..), object, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Either (isLeft)
import Hedgehog
import Invar.Tokenizer qualified as Tokenizer
import Numeric.Natural (Natural)

tokenizers :: Group
tokenizers =
    Group
        "Tokenizer descriptions and path consumption"
        [ ("a description derives its identity and its token space identity, and only a token space change moves the latter", once identities)
        , ("a description with another format, a missing or extra field, unordered or repeated added tokens or an empty vocabulary is refused", once malformed)
        , ("a token belongs to the base vocabulary or the added tokens, and an unassigned coordinate belongs to neither", once membership)
        , ("a target consumes a source path only when every source identifier means the same there, and may add identifiers outside the source's", once embedding)
        , ("a consumed path lies inside the target's input embedding and output distribution", once bounds)
        ]
  where
    once = withTests 1 . property

data Layout = Layout {size :: Natural, vocabulary :: Char, added :: [(Natural, String)], decoder :: Char, text :: Char}

base :: Layout
base = Layout 10 '1' [(10, "<eos>"), (11, "<think>")] '2' '3'

value :: Layout -> Value
value chosen =
    object
        [ "format" .= ("invar-tokenizer-v2" :: String)
        , "token_space" .= object ["format" .= ("invar-token-space-v1" :: String), "vocabulary" .= object ["size" .= size chosen, "sha256" .= replicate 64 (vocabulary chosen)], "added" .= [object ["id" .= token, "content" .= content] | (token, content) <- added chosen], "decoder" .= replicate 64 (decoder chosen)]
        , "text" .= replicate 64 (text chosen)
        , "execution" .= replicate 64 '4'
        ]

described :: Layout -> PropertyT IO Tokenizer.Description
described = evalEither . Tokenizer.describe . value

identities :: PropertyT IO ()
identities = do
    original <- described base
    template <- described base {text = '5'}
    extended <- described base {added = added base ++ [(12, "<audio>")]}
    Tokenizer.identity original /== Tokenizer.identity template
    Tokenizer.spaceIdentity (Tokenizer.space original) === Tokenizer.spaceIdentity (Tokenizer.space template)
    Tokenizer.spaceIdentity (Tokenizer.space original) /== Tokenizer.spaceIdentity (Tokenizer.space extended)
    again <- described base
    Tokenizer.identity again === Tokenizer.identity original

malformed :: PropertyT IO ()
malformed = do
    let edit key replacement (Object fields) = Object (Fields.insert key replacement fields)
        edit _ _ other = other
        space chosen = case value chosen of
            Object fields -> Fields.lookup "token_space" fields
            _ -> Nothing
        refused =
            [ edit "format" (String "invar-tokenizer-operation-v1") (value base)
            , edit "extra" (String "field") (value base)
            , case value base of
                Object fields -> Object (Fields.delete "execution" fields)
                other -> other
            , edit "text" (String "TEXT") (value base)
            , maybe Null (\inner -> edit "token_space" (edit "format" (String "invar-token-space-v0") inner) (value base)) (space base)
            ]
    assert (all (isLeft . Tokenizer.describe) refused)
    assert (all (isLeft . Tokenizer.describe . value) [base {added = [(11, "<think>"), (10, "<eos>")]}, base {added = [(10, "<eos>"), (10, "<eos>")]}, base {added = [(10, "")]}, base {size = 0}])

membership :: PropertyT IO ()
membership = do
    chosen <- Tokenizer.space <$> described base {added = [(3, "<pad>"), (15, "<eos>")]}
    map (Tokenizer.member chosen) [0, 3, 9, 15] === [True, True, True, True]
    map (Tokenizer.member chosen) [10, 14, 16] === [False, False, False]

embedding :: PropertyT IO ()
embedding = do
    source <- Tokenizer.space <$> described base
    let consumed target = Tokenizer.consume "trajectory" source target (Tokenizer.Coordinates 64 64) ([0, 1], [2, 11])
    accepted <- traverse described [base, base {text = '5'}, base {added = added base ++ [(12, "<audio>"), (20, "<tts>")]}]
    assert (not (any (isLeft . consumed) accepted))
    refused <-
        traverse
            described
            [ base {added = [(10, "<eos>")]}
            , base {added = [(10, "<eos>"), (11, "<other>")]}
            , base {added = [(4, "<pad>"), (10, "<eos>"), (11, "<think>")]}
            , base {vocabulary = '6'}
            , base {size = 11, added = [(11, "<think>")]}
            , base {decoder = '7'}
            ]
    assert (all (isLeft . consumed) refused)
    smaller <- described base {added = [(10, "<eos>")]}
    larger <- Tokenizer.space <$> described base {added = added base ++ [(12, "<audio>")]}
    assert (isLeft (Tokenizer.consume "trajectory" larger smaller (Tokenizer.Coordinates 64 64) ([0], [1])))

bounds :: PropertyT IO ()
bounds = do
    chosen <- described base
    let source = Tokenizer.space chosen
        consumed = Tokenizer.consume "trajectory" source chosen
    assert (not (isLeft (consumed (Tokenizer.Coordinates 12 12) ([0, 9], [10, 11]))))
    assert (not (isLeft (consumed (Tokenizer.Coordinates 11 12) ([0, 9], [10, 11]))))
    assert (isLeft (consumed (Tokenizer.Coordinates 10 12) ([0, 9], [10, 11])))
    assert (isLeft (consumed (Tokenizer.Coordinates 12 11) ([0, 9], [10, 11])))
    assert (isLeft (consumed (Tokenizer.Coordinates 9 12) ([0, 9], [1])))
