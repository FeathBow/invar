module Invar.Policy (identity) where

import Control.Exception (bracket)
import Control.Monad (foldM, unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Bits ((.&.))
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Invar.Artifact qualified as Artifact
import Invar.Policy.Encoding qualified as Encoding
import Invar.Policy.Header qualified as Header
import System.IO (Handle, SeekMode (AbsoluteSeek), hClose, hFileSize, hSeek)

headerBytes :: Integer
headerBytes = 8

maximumHeader :: Integer
maximumHeader = 100000000

identity :: FilePath -> IO String
identity path = bracket (Artifact.open "Policy adapter" path) hClose inspect

inspect :: Handle -> IO String
inspect file = do
    size <- hFileSize file
    prefix <- exact file headerBytes
    let lengthHeader = littleEndian prefix
        start = headerBytes + lengthHeader
    when (lengthHeader > maximumHeader || start > size) (invalid "Invalid policy header length")
    encoded <- exact file lengthHeader
    tensors <- either invalid pure (Header.decode encoded (size - start))
    context <- foldM (hashTensor (file, start)) SHA256.init tensors
    pure (Artifact.hex (SHA256.finalize context))

hashTensor :: (Handle, Integer) -> SHA256.Ctx -> Header.Tensor -> IO SHA256.Ctx
hashTensor (file, start) context tensor = do
    hSeek file AbsoluteSeek (start + Header.begin tensor)
    consume (file, Header.end tensor - Header.begin tensor) (SHA256.update context (Encoding.metadata tensor))

consume :: (Handle, Integer) -> SHA256.Ctx -> IO SHA256.Ctx
consume (_, 0) context = pure context
consume (file, remaining) context = do
    let count = min remaining (fromIntegral Artifact.chunkSize)
    chunk <- exact file count
    unless (finite chunk) (invalid "Policy adapter contains a non-finite FP32 tensor")
    consume (file, remaining - count) (SHA256.update context chunk)

finite :: ByteString -> Bool
finite bytes = all finiteWord [0, wordSize .. Bytes.length bytes - wordSize]
  where
    wordSize = fromIntegral Header.fp32Bytes
    upperExponent = 0x7f
    lowerExponent = 0x80
    finiteWord offset = Bytes.index bytes (offset + wordSize - 1) .&. upperExponent /= upperExponent || Bytes.index bytes (offset + wordSize - 2) .&. lowerExponent /= lowerExponent

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
