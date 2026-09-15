{-# LANGUAGE ImportQualifiedPost #-}
{-# LANGUAGE OverloadedStrings #-}

module Main (main) where

import Data.Aeson (Object, Value, eitherDecode, encode, object, toJSON, withObject, (.:), (.=))
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString.Lazy.Char8 qualified as Bytes
import Data.Text (Text)
import GHC.Float (castDoubleToWord64, castWord64ToDouble)
import Invar.Learn.Advantage qualified as Advantage

main :: IO ()
main = do
    encoded <- Bytes.getContents
    supplied <- either fail pure (eitherDecode encoded :: Either String [Value])
    observed <- either fail pure (traverse (parseEither reference) supplied)
    Bytes.putStrLn (encode observed)

reference :: Value -> Parser Value
reference = withObject "numerical case" calculate
  where
    calculate fields = do
        kind <- fields .: "kind" :: Parser Text
        case kind of
            "sum" -> do
                values <- fields .: "values"
                pure (result (toJSON . castDoubleToWord64 <$> Advantage.sum64 (map castWord64ToDouble values)))
            "advantage" -> do
                delta <- castWord64ToDouble <$> fields .: "delta"
                supplied <- fields .: "rewards" >>= traverse reward
                pure (result (toJSON <$> Advantage.calculate delta supplied))
            _ -> fail "Unknown numerical case kind"

reward :: Object -> Parser Advantage.Reward
reward fields = Advantage.Reward <$> fields .: "sample" <*> fields .: "group" <*> (castWord64ToDouble <$> fields .: "bits")

result :: Either Advantage.Error Value -> Value
result (Left problem) = object ["accepted" .= False, "reason" .= show problem]
result (Right observed) = object ["accepted" .= True, "words" .= observed]
