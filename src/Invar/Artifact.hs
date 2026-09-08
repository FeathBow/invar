module Invar.Artifact (open, identity, hash, hex, chunkSize) where

import Control.Exception (bracket, bracketOnError)
import Control.Monad (unless)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Numeric (showHex)
import System.IO (Handle, hClose)
import System.Posix.Files (getFdStatus, isRegularFile)
import System.Posix.IO qualified as Posix

chunkSize :: Int
chunkSize = 64 * 1024

open :: String -> FilePath -> IO Handle
open label path = bracketOnError (Posix.openFd path Posix.ReadOnly flags) Posix.closeFd $ \file -> do
    status <- getFdStatus file
    unless (isRegularFile status) (ioError (userError (label ++ " is not a regular file")))
    Posix.fdToHandle file
  where
    flags = Posix.defaultFileFlags {Posix.nofollow = True, Posix.cloexec = True, Posix.nonBlock = True}

identity :: String -> FilePath -> IO String
identity label path = bracket (open label path) hClose (fmap hex . hash)

hash :: Handle -> IO ByteString
hash file = consume SHA256.init
  where
    consume !context = do
        chunk <- Bytes.hGetSome file chunkSize
        if Bytes.null chunk
            then pure (SHA256.finalize context)
            else consume (SHA256.update context chunk)

hex :: ByteString -> String
hex = concatMap octet . Bytes.unpack
  where
    octet byte = case showHex byte "" of
        [digit] -> ['0', digit]
        digits -> digits
