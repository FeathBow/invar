{-# LANGUAGE OverloadedStrings #-}

module Invar.Tokenizer (Description, Space, Coordinates (..), Consumption, describe, identity, space, spaceIdentity, member, coordinates, consume, fed, consumedCoordinates, consumptionValue) where

import Control.Monad (unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Object, Value, object, withObject, (.:), (.=))
import Data.Aeson.Types (Parser, parseEither)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as Text
import Invar.Artifact qualified as Artifact
import Invar.Canonical qualified as Canonical
import Invar.Json qualified as Json
import Numeric.Natural (Natural)

data Space = Space {size :: Natural, vocabulary :: String, added :: Map Natural Text, decoder :: String}
    deriving (Eq, Show)

data Description = Description {tokenSpace :: Space, text :: String, execution :: String}
    deriving (Eq, Show)

data Coordinates = Coordinates {input :: Natural, output :: Natural}
    deriving (Eq, Show)

data Consumption = Consumption {path :: String, source :: String, target :: String, bounds :: Coordinates}
    deriving (Eq, Show)

describe :: Value -> Either String Description
describe = parseEither (withObject "tokenizer description" description)

description :: Object -> Parser Description
description fields = do
    Json.fields ["format", "token_space", "text", "execution"] fields
    format fields descriptionFormat
    Description <$> (fields .: "token_space" >>= withObject "token space" spaceFields) <*> (fields .: "text" >>= Json.identity) <*> (fields .: "execution" >>= Json.identity)

spaceFields :: Object -> Parser Space
spaceFields fields = do
    Json.fields ["format", "vocabulary", "added", "decoder"] fields
    format fields spaceFormat
    (count, listed) <- fields .: "vocabulary" >>= withObject "base vocabulary" (\entry -> Json.fields ["size", "sha256"] entry >> (,) <$> entry .: "size" <*> (entry .: "sha256" >>= Json.identity))
    entries <- fields .: "added" >>= mapM (withObject "added token" (\entry -> Json.fields ["id", "content"] entry >> (,) <$> entry .: "id" <*> entry .: "content"))
    when (count == 0) (fail "A token space needs a nonempty base vocabulary")
    unless (and (zipWith (<) (map fst entries) (drop 1 (map fst entries)))) (fail "Added tokens must be listed once each in increasing identifier order")
    when (any (Text.null . snd) entries) (fail "Added token content must be nonempty")
    Space count listed (Map.fromAscList entries) <$> (fields .: "decoder" >>= Json.identity)

format :: Object -> Text -> Parser ()
format fields expected = do
    actual <- fields .: "format"
    unless (actual == expected) (fail ("Expected the format " ++ Text.unpack expected))

coordinates :: Value -> Either String Coordinates
coordinates = parseEither $ withObject "target token coordinates" $ \fields -> do
    Json.fields ["input", "output"] fields
    bounds' <- Coordinates <$> fields .: "input" <*> fields .: "output"
    when (input bounds' == 0 || output bounds' == 0) (fail "Target token coordinates must be nonempty")
    pure bounds'

identity :: Description -> String
identity = digest . descriptionValue

space :: Description -> Space
space = tokenSpace

spaceIdentity :: Space -> String
spaceIdentity = digest . spaceValue

member :: Space -> Natural -> Bool
member selected token = token < size selected || Map.member token (added selected)

-- The target may consume a path of the source when every source identifier means the same in the target, which may add
-- identifiers outside the source's, and the path lies inside the tensors the target actually loaded. Scoring feeds the
-- prefix and every response token but the last, which it only reads from the output distribution.
consume :: String -> Space -> Description -> Coordinates -> ([Natural], [Natural]) -> Either String Consumption
consume trajectory from to bounds' (prefix, response) = do
    embeds from (tokenSpace to)
    fed bounds' (prefix ++ take (length response - 1) response)
    unless (all (< output bounds') response) (Left "A response token lies outside the target's output distribution")
    pure (Consumption trajectory (spaceIdentity from) (identity to) bounds')

-- Every token the target feeds through its input embedding must lie inside it, including those an executor feeds beyond
-- what scoring needs.
fed :: Coordinates -> [Natural] -> Either String ()
fed bounds' tokens = unless (all (< input bounds') tokens) (Left "A path token lies outside the target's input embedding")

embeds :: Space -> Space -> Either String ()
embeds from to = do
    unless ((size from, vocabulary from) == (size to, vocabulary to)) (Left "The target's base vocabulary differs from the source's")
    unless (decoder from == decoder to) (Left "The target decodes tokens differently from the source")
    unless (Map.filterWithKey (\token _ -> member from token) (added to) == added from) (Left "The target interprets a source token identifier differently")

consumedCoordinates :: Consumption -> Coordinates
consumedCoordinates = bounds

consumptionValue :: Consumption -> Value
consumptionValue selected =
    object
        [ "relation" .= ("source token space embedded in the target's" :: Text)
        , "trajectory" .= path selected
        , "source_token_space" .= source selected
        , "target_tokenizer" .= target selected
        , "coordinates" .= object ["input" .= input (bounds selected), "output" .= output (bounds selected)]
        ]

descriptionFormat, spaceFormat :: Text
descriptionFormat = "invar-tokenizer-v2"
spaceFormat = "invar-token-space-v1"

descriptionValue :: Description -> Canonical.Value
descriptionValue selected = record [("format", Canonical.Text descriptionFormat), ("token_space", spaceValue (tokenSpace selected)), ("text", string (text selected)), ("execution", string (execution selected))]

spaceValue :: Space -> Canonical.Value
spaceValue selected =
    record
        [ ("format", Canonical.Text spaceFormat)
        , ("vocabulary", record [("size", natural (size selected)), ("sha256", string (vocabulary selected))])
        , ("added", Canonical.Array [record [("id", natural token), ("content", Canonical.Text content)] | (token, content) <- Map.toAscList (added selected)])
        , ("decoder", string (decoder selected))
        ]

digest :: Canonical.Value -> String
digest = Artifact.hex . SHA256.hash . Canonical.encode

record :: [(Text, Canonical.Value)] -> Canonical.Value
record = Canonical.Object . Map.fromList

string :: String -> Canonical.Value
string = Canonical.Text . Text.pack

natural :: Natural -> Canonical.Value
natural = Canonical.Integer . toInteger
