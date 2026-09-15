{-# LANGUAGE OverloadedStrings #-}

module Invar.Policy.File (File, withFile, tensors, metadata, identity, rawIdentity, seekTensor, exact, finite, equal, sameRepresentation, nonzero) where

import Control.Exception (bracket)
import Control.Monad (foldM, unless, when, (>=>))
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Bits ((.&.))
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Invar.Artifact qualified as Artifact
import Invar.Policy.Encoding qualified as Encoding
import Invar.Policy.Header qualified as Header
import System.IO (Handle, SeekMode (AbsoluteSeek), hClose, hFileSize, hSeek)

data File = File Handle Integer (Map Text Text) [Header.Tensor]

withFile :: FilePath -> (File -> IO value) -> IO value
withFile path action = bracket (Artifact.open "Tensor artifact" path) hClose (inspect >=> action)

tensors :: File -> [Header.Tensor]
tensors (File _ _ _ entries) = entries

metadata :: File -> Map Text Text
metadata (File _ _ attributes _) = attributes

headerBytes, maximumHeader :: Integer
headerBytes = 8
maximumHeader = 100000000

inspect :: Handle -> IO File
inspect file = do
    size <- hFileSize file
    prefix <- exact file headerBytes
    let lengthHeader = littleEndian prefix
        start = headerBytes + lengthHeader
    when (lengthHeader > maximumHeader || start > size) (invalid "Invalid policy header length")
    encoded <- exact file lengthHeader
    (attributes, entries) <- either invalid pure (Header.decode encoded (size - start))
    pure (File file start attributes entries)

identity :: File -> IO String
identity source = do
    encoding <- case Map.lookup "invar_policy" (metadata source) of
        Nothing -> pure Encoding.metadata
        Just "mlx-f32/v1" -> pure Encoding.mlxMetadata
        _ -> invalid "Unsupported policy tensor identity encoding"
    Artifact.hex . SHA256.finalize <$> foldM (hashTensor encoding) SHA256.init (tensors source)
  where
    hashTensor encoding context tensor = do
        file <- seekTensor source tensor
        consume (file, Header.end tensor - Header.begin tensor) (SHA256.update context (encoding tensor))

rawIdentity :: File -> IO String
rawIdentity (File file _ _ _) = hSeek file AbsoluteSeek 0 >> fmap Artifact.hex (Artifact.hash file)

seekTensor :: File -> Header.Tensor -> IO Handle
seekTensor (File file start _ _) tensor = hSeek file AbsoluteSeek (start + Header.begin tensor) >> pure file

equal :: (File, Header.Tensor) -> (File, Header.Tensor) -> IO Bool
equal (left, first) (right, second)
    | lengthBytes first /= lengthBytes second = do
        _ <- nonzero left first
        _ <- nonzero right second
        pure False
    | otherwise = do
        initial <- seekTensor left first
        changed <- seekTensor right second
        compareBytes (initial, changed) (lengthBytes first) True
  where
    lengthBytes tensor = Header.end tensor - Header.begin tensor
    compareBytes _ 0 same = pure same
    compareBytes handles@(initial, changed) remaining same = do
        let count = min remaining (fromIntegral Artifact.chunkSize)
        before <- exact initial count
        after <- exact changed count
        unless (finite before && finite after) (invalid "Tensor artifact contains non-finite FP32 words")
        compareBytes handles (remaining - count) (same && before == after)

consume :: (Handle, Integer) -> SHA256.Ctx -> IO SHA256.Ctx
consume (_, 0) context = pure context
consume (file, remaining) context = do
    let count = min remaining (fromIntegral Artifact.chunkSize)
    chunk <- exact file count
    unless (finite chunk) (invalid "Policy adapter contains a non-finite FP32 tensor")
    consume (file, remaining - count) (SHA256.update context chunk)

-- Retain descriptor offsets and metadata values while ignoring JSON key order
-- and padding in the serialized header. Every payload word is still checked.
sameRepresentation :: File -> File -> IO Bool
sameRepresentation first second = do
    let left = tensors first
        right = tensors second
    compared <- traverse (\(before, after) -> equal (first, before) (second, after)) (zip left right)
    mapM_ (nonzero first) (drop (length right) left)
    mapM_ (nonzero second) (drop (length left) right)
    pure (metadata first == metadata second && left == right && and compared)

finite :: ByteString -> Bool
finite bytes = all finiteWord [0, wordSize .. Bytes.length bytes - wordSize]
  where
    wordSize = fromIntegral Header.fp32Bytes
    upperExponent = 0x7f
    lowerExponent = 0x80
    finiteWord offset = Bytes.index bytes (offset + wordSize - 1) .&. upperExponent /= upperExponent || Bytes.index bytes (offset + wordSize - 2) .&. lowerExponent /= lowerExponent

nonzero :: File -> Header.Tensor -> IO Bool
nonzero source tensor = do
    file <- seekTensor source tensor
    loop file (Header.end tensor - Header.begin tensor) False
  where
    loop _ 0 !found = pure found
    loop file remaining !found = do
        let count = min remaining (fromIntegral Artifact.chunkSize)
        encoded <- exact file count
        unless (finite encoded) (invalid "Tensor artifact contains non-finite FP32 words")
        loop file (remaining - count) (found || any (nonzeroWord encoded) [0, wordSize .. Bytes.length encoded - wordSize])
    nonzeroWord encoded offset = any (\index -> Bytes.index encoded (offset + index) /= 0) [0 .. wordSize - 2] || Bytes.index encoded (offset + wordSize - 1) .&. magnitudeMask /= 0
    wordSize = fromIntegral Header.fp32Bytes
    magnitudeMask = 0x7f

exact :: Handle -> Integer -> IO ByteString
exact file count = do
    unless (count >= 0 && count <= fromIntegral (maxBound :: Int)) (invalid "Policy read length is not representable")
    bytes <- Bytes.hGet file (fromInteger count)
    unless (fromIntegral (Bytes.length bytes) == count) (invalid "Truncated policy artifact")
    pure bytes

littleEndian :: ByteString -> Integer
littleEndian = Bytes.foldr (\byte value -> fromIntegral byte + octetBase * value) 0
  where
    octetBase = 256

invalid :: String -> IO value
invalid = ioError . userError
