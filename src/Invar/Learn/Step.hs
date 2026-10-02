{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Step (Record (..), stages, reports, decode, apply) where

import Data.Aeson (Object, (.:))
import Data.Aeson.Types (Parser)
import Data.Text (Text)
import Data.Word (Word32)
import Invar.Learn.Stream qualified as S

data Record = Proximal Text [Word32] | Reference Text [Word32] | Current S.Current | Applied S.Applied
    deriving (Eq, Show)

stages :: [Text]
stages = reports ++ ["applied"]

reports :: [Text]
reports = ["proximal", "reference", "current"]

decode :: Object -> Parser (Maybe Record)
decode fields = do
    stage <- fields .: "stage"
    case stage :: Text of
        "proximal" -> Just <$> (Proximal <$> fields .: "sample" <*> fields .: "words")
        "reference" -> Just <$> (Reference <$> fields .: "sample" <*> fields .: "words")
        "current" -> Just . Current <$> (S.Current <$> fields .: "step" <*> fields .: "sample" <*> fields .: "words" <*> fields .: "observation" <*> fields .: "state")
        "applied" -> Just . Applied <$> (S.Applied <$> fields .: "step" <*> fields .: "before" <*> fields .: "after" <*> fields .: "consumed")
        _ -> pure Nothing

apply :: S.Stream -> Record -> Either S.Error (S.Stream, Maybe S.Reply)
apply stream recorded = case recorded of
    Proximal name words32 -> (,Nothing) <$> S.proximal stream name words32
    Reference name words32 -> (,Nothing) <$> S.reference stream name words32
    Current report -> fmap Just <$> S.current stream report
    Applied report -> (,Nothing) . fst <$> S.applied stream report
