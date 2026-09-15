{-# LANGUAGE OverloadedStrings #-}

module Invar.Measurement.Profile (Profile, admit, fingerprint) where

import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Object, Value (..))
import Data.Aeson.Decoding.ByteString (bsToTokens)
import Data.Aeson.Decoding.Tokens (Tokens (..))
import Data.Aeson.Key qualified as Key
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Data.ByteString.Builder (Builder, byteString, char8, integerDec, string8, toLazyByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Char (ord)
import Data.List (intersperse)
import Data.Map.Strict qualified as Map
import Data.Ratio ((%))
import Data.Text (Text)
import Data.Text qualified as Text
import GHC.Float (floatToDigits)
import Invar.Artifact qualified as Artifact
import Invar.Json qualified as Json
import Numeric (showHex)

newtype Profile = Profile ByteString

-- Historical fingerprints use Python's sorted, ASCII JSON serialization.
-- Numeric tokens retain integer/float distinctions and floating zero signs.
admit :: ByteString -> Either String Profile
admit encoded = do
    _ <- Json.decode encoded
    (canonical, _) <- item (space encoded)
    pure (Profile (Lazy.toStrict (toLazyByteString canonical)))

fingerprint :: Profile -> Object -> Either String String
fingerprint (Profile reported) metadataFields = do
    metadata <- traverse textField (Fields.toList metadataFields)
    let combined = Map.insert "profile" (byteString reported) (Map.fromList metadata)
    pure (Artifact.hex (SHA256.hash (Lazy.toStrict (toLazyByteString (record combined)))))
  where
    textField (key, String value) = pure (Key.toText key, quoted value)
    textField _ = Left "Expected textual model metadata for the numerical profile"

item :: ByteString -> Either String (Builder, ByteString)
item encoded = case bsToTokens encoded of
    TkText value rest -> pure (quoted value, space rest)
    TkNumber _ rest -> do
        let literal = Bytes.take (Bytes.length encoded - Bytes.length rest) encoded
        numeric <- number literal
        pure (numeric, space rest)
    TkLit _ rest -> pure (byteString (if Bytes.take 1 encoded == "t" then "true" else if Bytes.take 1 encoded == "f" then "false" else "null"), space rest)
    TkArrayOpen _ -> array [] (space (Bytes.drop 1 encoded))
    TkRecordOpen _ -> fields Map.empty (space (Bytes.drop 1 encoded))
    TkErr problem -> Left problem

array :: [Builder] -> ByteString -> Either String (Builder, ByteString)
array accumulated encoded
    | Bytes.take 1 encoded == "]" = pure (char8 '[' <> separated (reverse accumulated) <> char8 ']', space (Bytes.drop 1 encoded))
    | otherwise = do
        (value, remaining) <- item encoded
        case Bytes.uncons remaining of
            Just (',', rest) -> array (value : accumulated) (space rest)
            Just (']', _) -> array (value : accumulated) remaining
            _ -> Left "Incomplete numerical-profile array"

fields :: Map.Map Text Builder -> ByteString -> Either String (Builder, ByteString)
fields accumulated encoded
    | Bytes.take 1 encoded == "}" = pure (record accumulated, space (Bytes.drop 1 encoded))
    | otherwise = case bsToTokens encoded of
        TkText key afterKey -> case Bytes.uncons (space afterKey) of
            Just (':', afterColon) -> do
                (value, remaining) <- item (space afterColon)
                let updated = Map.insert key value accumulated
                case Bytes.uncons remaining of
                    Just (',', rest) -> fields updated (space rest)
                    Just ('}', _) -> fields updated remaining
                    _ -> Left "Incomplete numerical-profile object"
            _ -> Left "Missing numerical-profile field separator"
        _ -> Left "Expected a numerical-profile object key"

record :: Map.Map Text Builder -> Builder
record values = char8 '{' <> separated [quoted key <> string8 ": " <> value | (key, value) <- Map.toAscList values] <> char8 '}'

separated :: [Builder] -> Builder
separated = mconcat . intersperse (string8 ", ")

space :: ByteString -> ByteString
space = Bytes.dropWhile (`elem` (" \t\r\n" :: String))

number :: ByteString -> Either String Builder
number literal
    | Bytes.any (`elem` (".eE" :: String)) literal = Json.floatingAt [] literal >>= fmap string8 . floating
    | otherwise = case Bytes.readInteger literal of
        Just (value, rest) | Bytes.null (space rest) -> pure (integerDec value)
        _ -> Left "Expected an integral numerical-profile token"

floating :: Double -> Either String String
floating value
    | value == 0 = pure (sign ++ "0.0")
    | otherwise = do
        (decimal, position) <- shortest (abs value)
        pure (sign ++ magnitude decimal position)
  where
    sign = if value < 0 || isNegativeZero value then "-" else ""
    magnitude decimal position
        | position - 1 < -4 || position - 1 >= 16 = scientific decimal (position - 1)
        | position <= 0 = "0." ++ replicate (-position) '0' ++ decimal
        | position >= length decimal = decimal ++ replicate (position - length decimal) '0' ++ ".0"
        | otherwise = let (before, after) = splitAt position decimal in before ++ "." ++ after
    scientific [] _ = "0.0"
    scientific (first : rest) power = first : (if null rest then "" else '.' : rest) ++ "e" ++ (if power < 0 then "-" else "+") ++ padded 2 (show (abs power))

-- Include decimal rounding boundaries: for example, 1e23 round-trips to
-- its binary64 value even when floatToDigits emits a longer spelling.
shortest :: Double -> Either String (String, Int)
shortest value = do
    initial <- maybe (Left "No round-tripping numerical-profile decimal") pure (candidate (length digits))
    let (coefficient, power) = reduce (length digits - 1) initial
    pure (normalize coefficient power)
  where
    (digits, position) = floatToDigits 10 value
    exact = toRational value
    candidate precision =
        let power = position - precision
            scale = if power >= 0 then (10 ^ power) % 1 else 1 % (10 ^ (-power))
            target = exact / scale
            lower = floor target
            upper = lower + 1
            matches coefficient = (fromRational (fromInteger coefficient * scale) :: Double) == value
            preferred = if target - fromInteger lower < fromInteger upper - target || (target - fromInteger lower == fromInteger upper - target && even lower) then [lower, upper] else [upper, lower]
         in case filter matches preferred of
                first : _ -> Just (first, power)
                [] -> Nothing
    reduce precision previous
        | precision <= 0 = previous
        | otherwise = maybe previous (reduce (precision - 1)) (candidate precision)
    normalize coefficient power
        | coefficient `mod` 10 == 0 = normalize (coefficient `div` 10) (power + 1)
        | otherwise = let decimal = show coefficient in (decimal, length decimal + power)

quoted :: Text -> Builder
quoted value = char8 '"' <> Text.foldr (\character rest -> escaped character <> rest) mempty value <> char8 '"'

escaped :: Char -> Builder
escaped character = case character of
    '"' -> string8 "\\\""
    '\\' -> string8 "\\\\"
    '\b' -> string8 "\\b"
    '\f' -> string8 "\\f"
    '\n' -> string8 "\\n"
    '\r' -> string8 "\\r"
    '\t' -> string8 "\\t"
    _ | code >= 0x20 && code < 0x7f -> char8 character
    _ | code <= 0xffff -> unicode code
    _ -> let supplementary = code - 0x10000 in unicode (0xd800 + supplementary `div` 0x400) <> unicode (0xdc00 + supplementary `mod` 0x400)
  where
    code = ord character
    unicode point = string8 "\\u" <> string8 (padded 4 (showHex point ""))

padded :: Int -> String -> String
padded width text = replicate (max 0 (width - length text)) '0' ++ text
