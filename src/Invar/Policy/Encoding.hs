module Invar.Policy.Encoding (metadata, mlxMetadata) where

import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Char (ord)
import Data.List (intercalate)
import Data.Text qualified as Text
import Invar.Policy.Header (Tensor (..))
import Numeric (showHex)

metadata :: Tensor -> ByteString
metadata = typed "torch.float32"

mlxMetadata :: Tensor -> ByteString
mlxMetadata = typed "mlx.core.float32"

typed :: String -> Tensor -> ByteString
typed dtype entry = Bytes.pack ("[\"" ++ concatMap escape (Text.unpack (name entry)) ++ "\", \"" ++ dtype ++ "\", [" ++ intercalate ", " (map show (shape entry)) ++ "]]")

escape :: Char -> String
escape character = case lookup character escapes of
    Just escaped -> escaped
    Nothing
        | code < firstPrintable || code > lastAscii -> unicode code
        | otherwise -> [character]
  where
    code = ord character
    firstPrintable = 0x20
    lastAscii = 0x7e
    escapes = [('"', "\\\""), ('\\', "\\\\"), ('\b', "\\b"), ('\f', "\\f"), ('\n', "\\n"), ('\r', "\\r"), ('\t', "\\t")]

unicode :: Int -> String
unicode code
    | code < supplementary = unit code
    | otherwise = unit (highSurrogate + shifted `div` surrogateWidth) ++ unit (lowSurrogate + shifted `mod` surrogateWidth)
  where
    supplementary = 0x10000
    shifted = code - supplementary
    surrogateWidth = 0x400
    highSurrogate = 0xd800
    lowSurrogate = 0xdc00

unit :: Int -> String
unit code = "\\u" ++ replicate (hexWidth - length digits) '0' ++ digits
  where
    digits = showHex code ""
    hexWidth = 4
