module Invar.Canonical (Value (..), encode) where

import Data.ByteString (ByteString)
import Data.ByteString.Builder qualified as Builder
import Data.ByteString.Lazy qualified as Lazy
import Data.Char (ord)
import Data.List (sortOn)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.Encoding (encodeUtf8)

data Value = Null | Boolean Bool | Integer Integer | Text Text | Array [Value] | Object (Map Text Value)
    deriving (Eq, Show)

encode :: Value -> ByteString
encode = Lazy.toStrict . Builder.toLazyByteString . build

build :: Value -> Builder.Builder
build value = case value of
    Null -> Builder.string7 "null"
    Boolean True -> Builder.string7 "true"
    Boolean False -> Builder.string7 "false"
    Integer number -> Builder.integerDec number
    Text text -> quoted text
    Array values -> Builder.char7 '[' <> separated (map build values) <> Builder.char7 ']'
    Object fields -> Builder.char7 '{' <> separated [quoted key <> Builder.char7 ':' <> build field | (key, field) <- sortOn (encodeUtf8 . fst) (Map.toList fields)] <> Builder.char7 '}'

separated :: [Builder.Builder] -> Builder.Builder
separated [] = mempty
separated (first : rest) = first <> foldMap (Builder.char7 ',' <>) rest

quoted :: Text -> Builder.Builder
quoted text = Builder.char7 '"' <> Text.foldr ((<>) . escaped) mempty text <> Builder.char7 '"'

escaped :: Char -> Builder.Builder
escaped character
    | character == '"' = Builder.string7 "\\\""
    | character == '\\' = Builder.string7 "\\\\"
    | ord character < 0x20 = Builder.string7 "\\u00" <> Builder.word8HexFixed (fromIntegral (ord character))
    | otherwise = Builder.byteString (encodeUtf8 (Text.singleton character))
