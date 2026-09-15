{-# LANGUAGE ImportQualifiedPost #-}
{-# LANGUAGE OverloadedStrings #-}

module Main (main) where

import Data.Aeson (Object, Value, eitherDecode, encode, object, toJSON, withObject, (.:), (.=))
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString.Lazy.Char8 qualified as Bytes
import Data.Text (Text)
import GHC.Float (castWord64ToDouble)
import Invar.Learn.Objective qualified as Objective

main :: IO ()
main = do
    encoded <- Bytes.getContents
    supplied <- either fail pure (eitherDecode encoded :: Either String [Value])
    observed <- either fail pure (traverse (parseEither reference) supplied)
    Bytes.putStrLn (encode observed)

reference :: Value -> Parser Value
reference = withObject "scalar case" calculate
  where
    calculate fields = do
        kind <- fields .: "kind" :: Parser Text
        case kind of
            "mean" -> result . fmap toJSON . Objective.mean32 <$> fields .: "values"
            "tokens" -> do
                epsilon <- castWord64ToDouble <$> fields .: "epsilon"
                penalty <- castWord64ToDouble <$> fields .: "penalty"
                count <- fields .: "count"
                values <- fields .: "inputs" >>= traverse inputs
                pure (result (output <$> Objective.calculate (Objective.Profile epsilon penalty) count values))
            _ -> fail "Unknown scalar case kind"

inputs :: Object -> Parser Objective.Inputs
inputs fields = do
    current <- fields .: "current"
    proximal <- fields .: "proximal"
    behavior <- fields .: "behavior"
    fixed <- fields .: "reference"
    advantage <- fields .: "advantage"
    pure Objective.Inputs {Objective.current = current, Objective.proximal = proximal, Objective.behavior = behavior, Objective.fixed = fixed, Objective.advantage = advantage}

output :: [Objective.Output] -> Value
output values = object ["terms" .= map Objective.term values, "current_gradient" .= map Objective.gradient values, "reward_gradient" .= map Objective.rewardGradient values]

result :: Either Objective.Error Value -> Value
result (Left problem) = object ["accepted" .= False, "kind" .= kind problem, "reason" .= show problem]
  where
    kind Objective.InvalidInput {} = "invalid_input" :: Text
    kind Objective.NonFinite {} = "non_finite"
result (Right value) = object ["accepted" .= True, "words" .= value]
