module Invar.Journal (Journal, with, resume, inspect, append, transcript, entries) where

import Control.Concurrent.MVar (MVar, modifyMVar, modifyMVar_, newMVar, readMVar, withMVar)
import Control.Exception (bracket, finally, mask_, onException)
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
import System.IO.Error (catchIOError, isDoesNotExistError, isEOFError)
import System.Posix.Files (setFdSize)
import System.Posix.IO qualified as Posix
import System.Posix.IO.ByteString qualified as Raw
import System.Posix.Types (Fd)

data Journal = Journal (MVar (Maybe Fd)) (MVar (Maybe (Map Int (MVar (Maybe (Fd, Bool))))))

with :: FilePath -> Value -> (Journal -> IO value) -> IO value
with path declaration = bracket (create path declaration) close

create :: FilePath -> Value -> IO Journal
create path declaration = do
    file <- exclusive path
    (locked Posix.WriteLock file >> write file (line declaration) >> synchronizeEntry path file) `onException` Posix.closeFd file
    Journal <$> newMVar (Just file) <*> newMVar (Just Map.empty)

resume :: FilePath -> ([Object] -> IO (Either failure accepted)) -> (accepted -> Journal -> IO value) -> IO (Either failure value)
resume path admit action = bracket (existing path Posix.ReadWrite Posix.defaultFileFlags {Posix.append = True}) Posix.closeFd $ \file -> do
    locked Posix.WriteLock file
    encoded <- contents file
    recorded <- either (ioError . userError) pure (entries encoded)
    when (null recorded) (ioError (userError "The journal has no complete declaration"))
    admitted <- admit recorded
    case admitted of
        Left problem -> pure (Left problem)
        Right accepted -> do
            let complete = Bytes.length encoded - Bytes.length (Char.takeWhileEnd (/= '\n') encoded)
            when (complete /= Bytes.length encoded) (setFdSize file (fromIntegral complete) >> Store.synchronizeFile file)
            lock <- newMVar (Just file)
            open <- newMVar (Just Map.empty)
            let journal = Journal lock open
            Right <$> action accepted journal `finally` abandon journal

inspect :: FilePath -> ([Object] -> IO value) -> IO value
inspect path action = bracket (existing path Posix.ReadOnly Posix.defaultFileFlags) Posix.closeFd $ \file -> do
    locked Posix.ReadLock file
    recorded <- contents file >>= either (ioError . userError) pure . entries
    when (null recorded) (ioError (userError "The journal has no complete declaration"))
    action recorded

close :: Journal -> IO ()
close journal = abandon journal >>= mapM_ Posix.closeFd

abandon :: Journal -> IO (Maybe Fd)
abandon (Journal lock open) = do
    modifyMVar_ open (\current -> mapM_ (mapM_ (`modifyMVar_` shut)) current >> pure Nothing)
    modifyMVar lock (\held -> pure (Nothing, held))

append :: Journal -> Value -> IO ()
append (Journal lock open) entry = do
    readMVar open >>= mapM_ (mapM_ (`modifyMVar_` traverse synchronized))
    withMVar lock (maybe (ioError (userError "Journal is closed")) (\file -> mask_ (write file (line entry))))
  where
    synchronized (file, changed) = when changed (Store.synchronizeFile file) >> pure (file, False)

transcript :: Journal -> FilePath -> (Transcript.Outcome -> Value) -> IO Transcript.Transcript
transcript journal@(Journal _ open) path ending = mask_ $ do
    file <- exclusive path
    (held, key) <- registered file `onException` Posix.closeFd file
    let recorded encoded = modifyMVar_ held (maybe (ioError (userError "Transcript is closed")) (\(target, _) -> written target encoded >> pure (Just (target, True))))
        retire = modifyMVar_ open (\current -> modifyMVar_ held shut >> pure (Map.delete key <$> current))
    pure (Transcript.Transcript (recorded . (<> Char.singleton '\n')) recorded (\outcome -> append journal (ending outcome) `finally` retire))
  where
    registered file = do
        synchronizeEntry path file
        held <- newMVar (Just (file, False))
        key <- modifyMVar open (maybe (ioError (userError "Journal is closed")) (\current -> let next = maybe 0 ((+ 1) . fst) (Map.lookupMax current) in pure (Just (Map.insert next held current), next)))
        pure (held, key)

shut :: Maybe (Fd, Bool) -> IO (Maybe (Fd, Bool))
shut held = mapM_ (Posix.closeFd . fst) held >> pure Nothing

existing :: FilePath -> Posix.OpenMode -> Posix.OpenFileFlags -> IO Fd
existing path mode flags = Posix.openFd path mode flags `catchIOError` \problem -> ioError (if isDoesNotExistError problem then userError ("There is no run journal at " ++ path) else problem)

exclusive :: FilePath -> IO Fd
exclusive path = Posix.openFd path Posix.WriteOnly Posix.defaultFileFlags {Posix.append = True, Posix.exclusive = True, Posix.creat = Just 0o644}

locked :: Posix.LockRequest -> Fd -> IO ()
locked kind file = Posix.setLock file (kind, AbsoluteSeek, 0, 0) `catchIOError` const (ioError (userError "Another process holds the run journal"))

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
