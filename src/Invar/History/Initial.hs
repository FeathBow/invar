{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Initial (Source (..), Run (..), Random (..), Checked, admit, schema, diagnostics, describe, state) where

import Control.Monad (unless, when, (>=>))
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Object, Value (..), object, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Map.Strict (Map)
import Data.Text (Text)
import Data.Text qualified as Text
import Invar.Artifact qualified as Artifact
import Invar.History.Artifacts qualified as Artifacts
import Invar.Json qualified as Json
import Invar.Learn qualified as Learn
import Invar.Learn.Codec (Decoder)
import Invar.Learn.State qualified as State
import Numeric.Natural (Natural)

data Source = Provided | Executed Run ByteString
data Run = Run {seed :: Integer, exitCode :: Int}
data Random = Torch Natural | MLX
    deriving (Eq, Show)
data Checked = Checked (Map Text [Integer]) Value [Value] State.Initial

admit :: Decoder -> (Learn.Settings, FilePath, Random) -> Source -> IO Checked
admit decoder (settings, path, random) source = do
    (origin, records) <- either invalid pure (observation settings source)
    (parameters, initial, observed) <- Artifacts.initial decoder (settings, path)
    either invalid pure (parseEither (withObject "initial checkpoint" (\fields -> fields .: "state" >>= rng random)) initial)
    fields <- either invalid pure (parseEither (withObject "initial checkpoint" pure) initial)
    pure (Checked parameters (Object (Fields.insert "source" origin fields)) records observed)

rng :: Random -> Object -> Parser ()
rng (Torch expected) fields = do
    Json.fields ["steps", "cpu_rng_bytes", "cuda_rng_bytes"] fields
    vectors <- fields .: "cuda_rng_bytes" :: Parser [Integer]
    unless (fromIntegral (length vectors) == expected) (fail "Initial CUDA RNG inventory differs from the declaration")
rng MLX fields = do
    Json.fields ["steps", "mlx_rng_bytes"] fields
    vectors <- fields .: "mlx_rng_bytes" :: Parser [Integer]
    unless (vectors == [mlxKeyBytes]) (fail "Initial MLX RNG inventory differs from the declaration")
  where
    mlxKeyBytes = 8

observation :: Learn.Settings -> Source -> Either String (Value, [Value])
observation _ Provided = pure (object ["kind" .= ("provided" :: Text)], [])
observation settings (Executed run encoded) = do
    unless (exitCode run == 0) (Left "Initialization process did not exit successfully")
    unless ("\n" `Bytes.isSuffixOf` encoded) (Left "Incomplete final initialization observation line")
    records <- traverse (Json.decode >=> parseEither (withObject "initialization observation" pure)) (Bytes.lines encoded)
    let stages = ["loading", "profile", "load", "checkpoint", "initial"]
    unless (map (Fields.lookup "stage") records == map (Just . String) stages && not (any (Fields.member "phase") records)) (Left "Expected one ordered initialization loading, profile, load, checkpoint and initial report")
    case records of
        [_, profile, loaded, saved, initial] -> do
            parseEither (report settings run) initial
            mapM_ (parseEither measurement) [loaded, saved]
            mapM_ (\key -> parseEither (\fields -> fields .: key >>= \value -> when (Text.null value) (fail "Expected nonempty initialization model and revision")) profile) ["model", "revision"]
            pure (object ["kind" .= ("initializer" :: Text), "seed" .= seed run, "exit_code" .= exitCode run, "log_sha256" .= Artifact.hex (SHA256.hash encoded), "report" .= Object initial], map Object (take (length stages - 1) records))
        _ -> Left "Incomplete initialization observation"

report :: Learn.Settings -> Run -> Object -> Parser ()
report settings run fields = do
    Json.fields (["stage", "policy", "learner", "tokenizer", "base", "assembly", "seed", "optimizer_steps"] ++ ["scope" | Fields.member "scope" fields]) fields
    mapM_ identity [("policy", Learn.policy settings), ("learner", Learn.learner settings), ("tokenizer", Learn.tokenizer settings), ("base", Learn.base settings), ("assembly", Learn.assembly settings)]
    actualSeed <- fields .: "seed"
    steps <- fields .: "optimizer_steps" :: Parser Integer
    unless (actualSeed == seed run && steps == 0) (fail "Initialization seed or optimizer steps differ from the declaration")
  where
    identity (key, expected) = do
        actual <- fields .: key >>= Json.identity
        unless (actual == expected) (fail "Initialization report differs from the declared checkpoint materialization")

measurement :: Object -> Parser ()
measurement fields | Fields.member "allocator" fields = do
    Json.fields ["stage", "seconds", "allocator", "peak_active", "cache_end"] fields
    seconds <- fields .: "seconds" >>= Json.finite
    active <- fields .: "peak_active" :: Parser Integer
    cached <- fields .: "cache_end" :: Parser Integer
    unless (Fields.lookup "allocator" fields == Just (String "mlx") && seconds >= 0 && active >= 0 && cached >= 0) (fail "Invalid native initialization measurement")
measurement fields = do
    Json.fields ["stage", "seconds", "peak_allocated", "peak_reserved"] fields
    seconds <- fields .: "seconds" >>= Json.finite
    allocated <- fields .: "peak_allocated" :: Parser Integer
    reserved <- fields .: "peak_reserved" :: Parser Integer
    unless (seconds >= 0 && allocated >= 0 && reserved >= allocated) (fail "Invalid initialization measurement observation")

schema :: Checked -> Map Text [Integer]
schema (Checked parameters _ _ _) = parameters

diagnostics :: Checked -> [Value]
diagnostics (Checked _ _ records _) = records

describe :: Checked -> Value
describe (Checked _ observed _ _) = observed

state :: Checked -> State.Initial
state (Checked _ _ _ observed) = observed

invalid :: String -> IO value
invalid = ioError . userError
