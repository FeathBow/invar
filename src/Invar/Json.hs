module Invar.Json (decode, textField, fields, identity, finite, floatingAt, floatingArrayAt, decodeWithArrays) where

import Control.Monad (unless, when)
import Data.Aeson (Object, Value, eitherDecodeStrict, parseJSON)
import Data.Aeson.Decoding.ByteString (bsToTokens)
import Data.Aeson.Decoding.Tokens
import Data.Aeson.Key (Key)
import Data.Aeson.Key qualified as Key
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text (Text)
import Invar.Digest qualified as Digest

decode :: ByteString -> Either String Value
decode encoded = do
    _ <- value (bsToTokens encoded)
    eitherDecodeStrict encoded

textField :: Key -> ByteString -> Either String Text
textField key encoded = do
    remaining <- value (bsToTokens encoded)
    unless (Bytes.null (space remaining)) (Left "Unexpected data after the JSON record")
    selected <- select [key] encoded
    case bsToTokens selected of
        TkText result _ -> pure result
        _ -> Left "Expected a text JSON field"

fields :: [Key] -> Object -> Parser ()
fields expected actual = unless (Set.fromList expected == Set.fromList (Fields.keys actual)) (fail "Unexpected or missing JSON fields")

identity :: Value -> Parser String
identity encoded = do
    decoded <- parseJSON encoded
    unless (Digest.sha256 decoded) (fail "Expected a lowercase SHA-256 identity")
    pure decoded

finite :: Value -> Parser Double
finite encoded = do
    decoded <- parseJSON encoded
    when (isNaN decoded || isInfinite decoded) (fail "Expected a finite JSON number")
    pure decoded

floatingAt :: [Key] -> ByteString -> Either String Double
floatingAt path encoded = do
    _ <- decode encoded
    select path encoded >>= floating

floatingArrayAt :: [Key] -> ByteString -> Either String [Double]
floatingArrayAt path encoded = do
    _ <- decode encoded
    floatingArray path encoded

decodeWithArrays :: [[Key]] -> ByteString -> Either String (Value, [[Double]])
decodeWithArrays paths encoded = do
    decoded <- decode encoded
    (,) decoded <$> traverse (`floatingArray` encoded) paths

floatingArray :: [Key] -> ByteString -> Either String [Double]
floatingArray path encoded = do
    selected <- select path encoded >>= punctuation '['
    items selected
  where
    items content
        | Bytes.take 1 content == Bytes.singleton ']' = Right []
        | otherwise = do
            remaining <- value (bsToTokens content)
            number <- floating (Bytes.take (Bytes.length content - Bytes.length remaining) content)
            let rest = space remaining
            following <- if Bytes.take 1 rest == Bytes.singleton ']' then pure [] else punctuation ',' rest >>= items
            pure (number : following)

floating :: ByteString -> Either String Double
floating literal = do
    number <- decode literal >>= parseEither finite
    let floatingLiteral = Bytes.any (`elem` (".eE" :: String)) literal
    pure (if number == 0 && Bytes.take 1 literal == Bytes.singleton '-' && floatingLiteral then -0.0 else number)

select :: [Key] -> ByteString -> Either String ByteString
select [] encoded = Right (space encoded)
select (key : path) encoded = do
    content <- punctuation '{' encoded
    fieldBytes key content >>= select path

fieldBytes :: Key -> ByteString -> Either String ByteString
fieldBytes expected encoded = case bsToTokens (space encoded) of
    TkText name rest -> do
        content <- punctuation ':' rest
        remaining <- value (bsToTokens content)
        if Key.fromText name == expected
            then pure (Bytes.take (Bytes.length content - Bytes.length remaining) content)
            else punctuation ',' remaining >>= fieldBytes expected
    _ -> Left "Missing observed JSON number field"

punctuation :: Char -> ByteString -> Either String ByteString
punctuation expected encoded = case Bytes.uncons (space encoded) of
    Just (actual, remaining) | actual == expected -> Right (space remaining)
    _ -> Left "Expected an object field containing the observed JSON number"

space :: ByteString -> ByteString
space = Bytes.dropWhile (`elem` (" \n\r\t" :: String))

value :: Tokens rest String -> Either String rest
value (TkLit _ rest) = Right rest
value (TkText _ rest) = Right rest
value (TkNumber _ rest) = Right rest
value (TkArrayOpen entries) = array entries
value (TkRecordOpen entries) = record Set.empty entries
value (TkErr problem) = Left problem

array :: TkArray rest String -> Either String rest
array (TkItem next) = value next >>= array
array (TkArrayEnd rest) = Right rest
array (TkArrayErr problem) = Left problem

record :: Set Key -> TkRecord rest String -> Either String rest
record seen (TkPair key next) = do
    when (Set.member key seen) (Left "Duplicate JSON key")
    remaining <- value next
    record (Set.insert key seen) remaining
record _ (TkRecordEnd rest) = Right rest
record _ (TkRecordErr problem) = Left problem
