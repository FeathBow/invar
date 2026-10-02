module Invar.Journal (Journal, with, resume, append, store, entries) where

import Control.Concurrent.MVar (MVar, modifyMVar_, newMVar, withMVar)
import Control.Exception (bracket, finally, onException)
import Control.Monad (when)
import Data.Aeson (Object, Value, eitherDecodeStrict, encode)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as Char
import Data.ByteString.Lazy qualified as Lazy
import Invar.Store qualified as Store
import System.FilePath (takeDirectory)
import System.IO (SeekMode (AbsoluteSeek))
import System.IO.Error (catchIOError, isEOFError)
import System.Posix.Files (setFdSize)
import System.Posix.IO qualified as Posix
import System.Posix.IO.ByteString qualified as Raw
import System.Posix.Types (Fd)

newtype Journal = Journal (MVar (Maybe Fd))

with :: FilePath -> Value -> (Journal -> IO value) -> IO value
with path declaration = bracket (create path declaration) close

create :: FilePath -> Value -> IO Journal
create path declaration = do
    file <- exclusive path
    (locked file >> write file (line declaration) >> synchronizeEntry path file) `onException` Posix.closeFd file
    Journal <$> newMVar (Just file)

resume :: FilePath -> ([Object] -> Journal -> IO value) -> IO value
resume path action = bracket (Posix.openFd path Posix.ReadWrite Posix.defaultFileFlags {Posix.append = True}) Posix.closeFd $ \file -> do
    locked file
    encoded <- contents file
    recorded <- either (ioError . userError) pure (entries encoded)
    when (null recorded) (ioError (userError "The journal has no complete declaration"))
    let complete = Bytes.length encoded - Bytes.length (Char.takeWhileEnd (/= '\n') encoded)
    when (complete /= Bytes.length encoded) (setFdSize file (fromIntegral complete) >> Store.synchronizeFile file)
    lock <- newMVar (Just file)
    action recorded (Journal lock) `finally` modifyMVar_ lock (const (pure Nothing))

close :: Journal -> IO ()
close (Journal lock) = modifyMVar_ lock (\held -> mapM_ Posix.closeFd held >> pure Nothing)

append :: Journal -> Value -> IO ()
append (Journal lock) entry = withMVar lock (maybe (ioError (userError "Journal is closed")) (\file -> write file (line entry)))

store :: FilePath -> ByteString -> IO ()
store path encoded = bracket (exclusive path) Posix.closeFd $ \file -> do
    write file encoded
    synchronizeEntry path file

exclusive :: FilePath -> IO Fd
exclusive path = Posix.openFd path Posix.WriteOnly Posix.defaultFileFlags {Posix.append = True, Posix.exclusive = True, Posix.creat = Just 0o644}

locked :: Fd -> IO ()
locked file = Posix.setLock file (Posix.WriteLock, AbsoluteSeek, 0, 0) `catchIOError` const (ioError (userError "Another process holds the run journal"))

contents :: Fd -> IO ByteString
contents file = go []
  where
    go chunks = do
        chunk <- (Just <$> Raw.fdRead file 65536) `catchIOError` \problem -> if isEOFError problem then pure Nothing else ioError problem
        maybe (pure (Bytes.concat (reverse chunks))) (go . (: chunks)) chunk

line :: Value -> ByteString
line entry = Lazy.toStrict (encode entry) <> Char.singleton '\n'

write :: Fd -> ByteString -> IO ()
write file remaining
    | Bytes.null remaining = Store.synchronizeFile file
    | otherwise = do
        written <- Raw.fdWrite file remaining
        write file (Bytes.drop (fromIntegral written) remaining)

synchronizeEntry :: FilePath -> Fd -> IO ()
synchronizeEntry path file = bracket (Posix.openFd (takeDirectory path) Posix.ReadOnly Posix.defaultFileFlags) Posix.closeFd (`Store.synchronizeDirectory` file)

entries :: ByteString -> Either String [Object]
entries encoded
    | Bytes.null encoded = Right []
    | otherwise = traverse eitherDecodeStrict (init (Char.split '\n' encoded))
