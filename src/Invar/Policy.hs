{-# LANGUAGE OverloadedStrings #-}

module Invar.Policy (identity, Description, describe, model, revision, adapter, tokenizer, base, assembly, bindings, successor, encodeDescription, decodeDescription, readDescription, stageDescription) where

import Control.Exception (bracket, bracketOnError)
import Control.Monad (unless, void, when, (>=>))
import Data.Aeson (Value (String), encode, object, withObject, (.:), (.=))
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Text qualified as Text
import Invar.Artifact qualified as Artifact
import Invar.Json qualified as Json
import Invar.Policy.Description (Description)
import Invar.Policy.Description qualified as Description
import Invar.Policy.File qualified as File
import System.IO (hClose)
import System.Posix.IO qualified as Posix

identity :: FilePath -> IO String
identity path = File.withFile path File.identity

describe :: (String, String) -> (String, String, String, String) -> Either String Description
describe (name, version) (policy, operation, frozen, profile) = do
    mapM_ nonempty [name, version]
    mapM_ validIdentity [policy, operation, frozen, profile]
    pure (Description.Description name version policy operation frozen profile)
  where
    nonempty value = when (null value || '\0' `elem` value) (Left "Expected nonempty model and revision text without NUL")

successor :: String -> Description -> Either String Description
successor changed selected = do
    validIdentity changed
    pure selected {Description.adapter = changed}

model, revision, adapter, tokenizer, base, assembly :: Description -> String
model = Description.model
revision = Description.revision
adapter = Description.adapter
tokenizer = Description.tokenizer
base = Description.base
assembly = Description.assembly

bindings :: Description -> (String, String, String, String)
bindings = Description.bindings

validIdentity :: String -> Either String ()
validIdentity value = void (parseEither Json.identity (String (Text.pack value)))

encodeDescription :: Description -> ByteString
encodeDescription selected = Lazy.toStrict (encode value) <> "\n"
  where
    value = object ["format" .= String "invar-policy-v1", "model" .= model selected, "revision" .= revision selected, "adapter" .= adapter selected, "tokenizer" .= tokenizer selected, "base" .= base selected, "assembly" .= assembly selected]

decodeDescription :: ByteString -> Either String Description
decodeDescription encoded = Json.decode encoded >>= parseEither parse
  where
    parse = withObject "immutable inference policy" $ \fields -> do
        Json.fields ["format", "model", "revision", "adapter", "tokenizer", "base", "assembly"] fields
        format <- fields .: "format"
        unless (format == ("invar-policy-v1" :: String)) (fail "Unknown inference policy description format")
        source <- (,) <$> fields .: "model" <*> fields .: "revision"
        inputs <- (,,,) <$> fields .: "adapter" <*> fields .: "tokenizer" <*> fields .: "base" <*> fields .: "assembly"
        either fail pure (describe source inputs)

readDescription :: FilePath -> IO Description
readDescription path = bracket (Artifact.open "Policy description" path) hClose (Bytes.hGetContents >=> either (ioError . userError) pure . decodeDescription)

-- Staging is exclusive; durability belongs to the checkpoint store operation.
stageDescription :: FilePath -> Description -> IO ()
stageDescription path selected = bracket acquire hClose (\file -> Bytes.hPut file (encodeDescription selected))
  where
    acquire = bracketOnError (Posix.openFd path Posix.WriteOnly flags) Posix.closeFd Posix.fdToHandle
    flags = Posix.defaultFileFlags {Posix.creat = Just 0o644, Posix.exclusive = True, Posix.nofollow = True, Posix.cloexec = True}
