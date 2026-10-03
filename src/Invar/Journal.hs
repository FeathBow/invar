module Invar.Journal (Journal, with, resume, append, store, transcript, entries) where

import Control.Concurrent.MVar (MVar, modifyMVar, modifyMVar_, newMVar, readMVar, withMVar)
import Control.Exception (bracket, finally, onException)
import Control.Monad (when)
import Data.Aeson (Object, Value, eitherDecodeStrict, encode)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as Char
import Data.ByteString.Lazy qualified as Lazy
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Invar.Store qualified as Store
import Invar.Transcript qualified as Transcript
import System.FilePath (takeDirectory)
import System.IO (SeekMode (AbsoluteSeek))
import System.IO.Error (catchIOError, isEOFError)
import System.Posix.Files (setFdSize)
import System.Posix.IO qualified as Posix
import System.Posix.IO.ByteString qualified as Raw
import System.Posix.Types (Fd)

data Journal = Journal (MVar (Maybe Fd)) (MVar (Map Int (MVar (Maybe (Fd, Bool)))))

with :: FilePath -> Value -> (Journal -> IO value) -> IO value
with path declaration = bracket (create path declaration) close

create :: FilePath -> Value -> IO Journal
create path declaration = do
    file <- exclusive path
    (locked file >> write file (line declaration) >> synchronizeEntry path file) `onException` Posix.closeFd file
    Journal <$> newMVar (Just file) <*> newMVar Map.empty

resume :: FilePath -> ([Object] -> Journal -> IO value) -> IO value
resume path action = bracket (Posix.openFd path Posix.ReadWrite Posix.defaultFileFlags {Posix.append = True}) Posix.closeFd $ \file -> do
    locked file
    encoded <- contents file
    recorded <- either (ioError . userError) pure (entries encoded)
    when (null recorded) (ioError (userError "The journal has no complete declaration"))
    let complete = Bytes.length encoded - Bytes.length (Char.takeWhileEnd (/= '\n') encoded)
    when (complete /= Bytes.length encoded) (setFdSize file (fromIntegral complete) >> Store.synchronizeFile file)
    lock <- newMVar (Just file)
    open <- newMVar Map.empty
    let journal = Journal lock open
    action recorded journal `finally` abandon journal

close :: Journal -> IO ()
close journal = abandon journal >>= mapM_ Posix.closeFd

abandon :: Journal -> IO (Maybe Fd)
abandon (Journal lock open) = do
    readMVar open >>= mapM_ (`modifyMVar_` (\held -> mapM_ (Posix.closeFd . fst) held >> pure Nothing))
    modifyMVar lock (\held -> pure (Nothing, held))

append :: Journal -> Value -> IO ()
append (Journal lock open) entry = do
    readMVar open >>= mapM_ (`modifyMVar_` traverse synchronized)
    withMVar lock (maybe (ioError (userError "Journal is closed")) (\file -> write file (line entry)))
  where
    synchronized (file, changed) = when changed (Store.synchronizeFile file) >> pure (file, False)

transcript :: Journal -> FilePath -> (Transcript.Outcome -> Value) -> IO Transcript.Transcript
transcript journal@(Journal _ open) path ending = do
    file <- exclusive path
    synchronizeEntry path file `onException` Posix.closeFd file
    held <- newMVar (Just (file, False))
    key <- modifyMVar open (\current -> let next = maybe 0 ((+ 1) . fst) (Map.lookupMax current) in pure (Map.insert next held current, next))
    let recorded encoded = modifyMVar_ held (maybe (ioError (userError "Transcript is closed")) (\(target, _) -> written target (encoded <> Char.singleton '\n') >> pure (Just (target, True))))
        finished outcome = do
            append journal (ending outcome)
            modifyMVar_ open (pure . Map.delete key)
            modifyMVar_ held (\current -> mapM_ (Posix.closeFd . fst) current >> pure Nothing)
    pure (Transcript.Transcript recorded finished)

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
write file encoded = written file encoded >> Store.synchronizeFile file

written :: Fd -> ByteString -> IO ()
written file remaining
    | Bytes.null remaining = pure ()
    | otherwise = do
        count <- Raw.fdWrite file remaining
        written file (Bytes.drop (fromIntegral count) remaining)

synchronizeEntry :: FilePath -> Fd -> IO ()
synchronizeEntry path file = bracket (Posix.openFd (takeDirectory path) Posix.ReadOnly Posix.defaultFileFlags) Posix.closeFd (`Store.synchronizeDirectory` file)

entries :: ByteString -> Either String [Object]
entries encoded
    | Bytes.null encoded = Right []
    | otherwise = traverse eitherDecodeStrict (init (Char.split '\n' encoded))
