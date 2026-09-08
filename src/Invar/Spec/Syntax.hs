{-# LANGUAGE Safe #-}

module Invar.Spec.Syntax (Term (..), parse, render, tagged) where

import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Char (isSpace)
import Text.ParserCombinators.ReadP

data Term = Bare String | Quoted String | List [Term]
    deriving (Eq, Show)

parse :: ByteString -> Either String Term
parse bytes
    | Bytes.any (> '\127') bytes = Left "Program text must be ASCII; use quoted escapes for Unicode names"
    | otherwise = case readP_to_S (skipSpaces *> term <* skipSpaces <* eof) (Bytes.unpack bytes) of
        [(value, "")] -> Right value
        _ -> Left "Invalid program text"

term :: ReadP Term
term = quoted <++ list <++ bare
  where
    quoted = do
        remaining <- look
        case remaining of
            '"' : _ -> Quoted <$> readS_to_P reads
            _ -> pfail
    list = List <$> between (char '(' *> skipSpaces) (char ')') (many (term <* skipSpaces))
    bare = Bare <$> munch1 (\c -> not (isSpace c) && c `notElem` ['(', ')', '"', '\\'])

render :: Term -> ByteString
render = Bytes.pack . write
  where
    write (Bare name) = name
    write (Quoted name) = show name
    write (List values) = "(" ++ unwords (map write values) ++ ")"

tagged :: String -> [Term] -> Term
tagged name values = List (Bare name : values)
