{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Codec (Decoder (..), Session, withSession, decode, chunk, consume, equal) where

import Control.Monad (unless)
import Data.Aeson (Value, encode, object, withObject, (.:), (.=))
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Set qualified as Set
import Data.Text (Text)
import Invar.Artifact qualified as Artifact
import Invar.Json qualified as Json
import Invar.Learn.Native qualified as Native
import Invar.Policy.File qualified as File
import System.Exit (ExitCode (ExitSuccess))
import System.IO (Handle, hClose, hFlush, hIsEOF, stdin, stdout)
import System.Process (CreateProcess (..), StdStream (CreatePipe, Inherit), proc, waitForProcess, withCreateProcess)

data Decoder = External FilePath FilePath | Standard

data Session = Session Handle Handle

withSession :: Decoder -> (Session -> IO value) -> IO value
withSession Standard action = do
    let session = Session stdout stdin
    result <- action session
    release session
    pure result
withSession (External executable script) action = do
    let command = (proc executable ["-B", script]) {std_in = CreatePipe, std_out = CreatePipe, std_err = Inherit}
    withCreateProcess command $ \incoming outgoing _ child -> case (incoming, outgoing) of
        (Just writer, Just reader) -> do
            result <- action (Session writer reader)
            hClose writer
            ended <- hIsEOF reader
            unless ended (invalid "Output follows the final native codec response")
            status <- waitForProcess child
            unless (status == ExitSuccess) (invalid ("Native codec process failed: " ++ show status))
            pure result
        _ -> invalid "Native codec pipes were not created"

release :: Session -> IO ()
release session@(Session _ reader) = do
    send session (object ["codec" .= ("release" :: Text)])
    encoded <- Bytes.hGetLine reader
    acknowledged <- either invalid pure (Json.decode encoded)
    unless (acknowledged == object ["released" .= True]) (invalid "Native codec did not release its scoped snapshots")

decode :: Session -> (FilePath, String) -> IO Native.Value
decode session@(Session _ reader) (path, expected) = do
    actual <- Artifact.identity "Learner checkpoint" path
    unless (actual == expected) (invalid "Learner file differs from its reported digest")
    send session (object ["codec" .= ("decode" :: Text), "path" .= path])
    encoded <- Bytes.hGetLine reader
    value <- either invalid pure (Json.decode encoded >>= parseEither (document actual))
    let tensors = Native.tensorValues value
        identities = map Native.index tensors
    unless (length identities == Set.size (Set.fromList identities)) (invalid "Duplicate native tensor references")
    pure value

document :: String -> Value -> Parser Native.Value
document expected = withObject "native checkpoint description" $ \fields -> do
    Json.fields ["format", "source_sha256", "byte_order", "value"] fields
    format <- fields .: "format"
    unless (format == ("invar-native-checkpoint/v1" :: Text)) (fail "Unknown native checkpoint codec format")
    digest <- fields .: "source_sha256" >>= Json.identity
    unless (digest == expected) (fail "Decoded learner snapshot differs from the supplied file")
    byteOrder <- fields .: "byte_order"
    unless (byteOrder == ("little" :: Text)) (fail "Native codec tensors must use little-endian bytes")
    fields .: "value" >>= Native.parse

send :: Session -> Value -> IO ()
send (Session writer _) value = Bytes.hPutStrLn writer (Lazy.toStrict (encode value)) >> hFlush writer

chunk :: Session -> Native.Tensor -> (Integer, Integer) -> IO ByteString
chunk session@(Session _ reader) tensor (offset, count) = do
    unless (offset >= 0 && count >= 0 && offset + count <= Native.size tensor) (invalid "Native tensor request exceeds its declared data")
    send session (object ["codec" .= ("tensor" :: Text), "index" .= Native.index tensor, "offset" .= offset, "count" .= count])
    File.exact reader count

consume :: (Session, Native.Tensor) -> (ByteString -> IO ()) -> IO ()
consume (session, tensor) check = loop 0
  where
    loop offset
        | offset == Native.size tensor = pure ()
        | otherwise = do
            let count = min (Native.size tensor - offset) (fromIntegral Artifact.chunkSize)
            chunk session tensor (offset, count) >>= check
            loop (offset + count)

equal :: Session -> (Native.Tensor, Native.Tensor) -> IO Bool
equal session (left, right)
    | Native.size left /= Native.size right = pure False
    | otherwise = loop 0 True
  where
    loop offset same
        | offset == Native.size left = pure same
        | otherwise = do
            let count = min (Native.size left - offset) (fromIntegral Artifact.chunkSize)
            first <- chunk session left (offset, count)
            second <- chunk session right (offset, count)
            loop (offset + count) (same && first == second)

invalid :: String -> IO value
invalid = ioError . userError
