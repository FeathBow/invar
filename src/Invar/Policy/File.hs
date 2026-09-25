{-# LANGUAGE OverloadedStrings #-}

module Invar.Policy.File (File, Scan (..), withFile, tensors, metadata, identity, identityScan, scan, rawIdentityScan, seekTensor, exact, finite, nonzero) where

import Control.Exception (bracket)
import Control.Monad (foldM, unless, when, (>=>))
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Bits ((.&.))
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.List (sortOn)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Invar.Artifact qualified as Artifact
import Invar.Policy.Encoding qualified as Encoding
import Invar.Policy.Header qualified as Header
import System.IO (Handle, SeekMode (AbsoluteSeek), hClose, hFileSize, hSeek)

data File = File Handle Integer (Map Text Text) [Header.Tensor]

data Scan = Scan {scanName :: Text, scanShape :: [Integer], scanDigest :: String, scanNonzero :: Bool}
    deriving (Eq, Show)

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
identity = fmap fst . identityScan

identityScan :: File -> IO (String, [Scan])
identityScan source = do
    encoding <- case Map.lookup "invar_policy" (metadata source) of
        Nothing -> pure Encoding.metadata
        Just "mlx-f32/v1" -> pure Encoding.mlxMetadata
        _ -> invalid "Unsupported policy tensor identity encoding"
    (context, scans) <- foldM (hashTensor encoding) (SHA256.init, []) (tensors source)
    pure (Artifact.hex (SHA256.finalize context), reverse scans)
  where
    hashTensor encoding (context, scans) tensor = do
        (updated, scanned) <- scanTensor source tensor (SHA256.update context (encoding tensor))
        pure (updated, scanned : scans)

scan :: File -> IO [Scan]
scan source = traverse (\tensor -> snd <$> scanTensor source tensor SHA256.init) (tensors source)

scanTensor :: File -> Header.Tensor -> SHA256.Ctx -> IO (SHA256.Ctx, Scan)
scanTensor source tensor outer = do
    file <- seekTensor source tensor
    loop file (Header.end tensor - Header.begin tensor) (outer, SHA256.init, False)
  where
    loop _ 0 (context, own, found) = pure (context, Scan (Header.name tensor) (Header.shape tensor) (Artifact.hex (SHA256.finalize own)) found)
    loop file remaining (context, own, found) = do
        let count = min remaining (fromIntegral Artifact.chunkSize)
        chunk <- exact file count
        unless (finite chunk) (invalid "Tensor artifact contains non-finite FP32 words")
        loop file (remaining - count) (SHA256.update context chunk, SHA256.update own chunk, found || nonzeroBytes chunk)

rawIdentityScan :: File -> IO (String, [Scan])
rawIdentityScan source@(File file start _ entries) = do
    size <- hFileSize file
    hSeek file AbsoluteSeek 0
    header <- exact file start
    (context, position, scans) <- foldM step (SHA256.update SHA256.init header, 0, Map.empty) (sortOn Header.begin entries)
    hSeek file AbsoluteSeek (start + position)
    trailing <- exact file (size - start - position)
    let ordered = [scans Map.! Header.name tensor | tensor <- entries]
    pure (Artifact.hex (SHA256.finalize (SHA256.update context trailing)), ordered)
  where
    step (context, position, scans) tensor = do
        hSeek file AbsoluteSeek (start + position)
        gap <- exact file (Header.begin tensor - position)
        (updated, scanned) <- scanTensor source tensor (SHA256.update context gap)
        pure (updated, Header.end tensor, Map.insert (Header.name tensor) scanned scans)

seekTensor :: File -> Header.Tensor -> IO Handle
seekTensor (File file start _ _) tensor = hSeek file AbsoluteSeek (start + Header.begin tensor) >> pure file

finite :: ByteString -> Bool
finite bytes = all finiteWord [0, wordSize .. Bytes.length bytes - wordSize]
  where
    wordSize = fromIntegral Header.fp32Bytes
    upperExponent = 0x7f
    lowerExponent = 0x80
    finiteWord offset = Bytes.index bytes (offset + wordSize - 1) .&. upperExponent /= upperExponent || Bytes.index bytes (offset + wordSize - 2) .&. lowerExponent /= lowerExponent

nonzero :: File -> Header.Tensor -> IO Bool
nonzero source tensor = scanNonzero . snd <$> scanTensor source tensor SHA256.init

nonzeroBytes :: ByteString -> Bool
nonzeroBytes encoded = any nonzeroWord [0, wordSize .. Bytes.length encoded - wordSize]
  where
    nonzeroWord offset = any (\index -> Bytes.index encoded (offset + index) /= 0) [0 .. wordSize - 2] || Bytes.index encoded (offset + wordSize - 1) .&. magnitudeMask /= 0
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
