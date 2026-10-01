module Invar.Journal (Journal, with, append, store, entries) where

import Control.Concurrent.MVar (MVar, modifyMVar_, newMVar, withMVar)
import Control.Exception (bracket, onException)
import Data.Aeson (Object, Value, eitherDecodeStrict, encode)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as Char
import Data.ByteString.Lazy qualified as Lazy
import Invar.Store qualified as Store
import System.FilePath (takeDirectory)
import System.Posix.IO qualified as Posix
import System.Posix.IO.ByteString qualified as Raw
import System.Posix.Types (Fd)

newtype Journal = Journal (MVar (Maybe Fd))

with :: FilePath -> Value -> (Journal -> IO value) -> IO value
with path declaration = bracket (create path declaration) close

create :: FilePath -> Value -> IO Journal
create path declaration = do
    file <- exclusive path
    (write file (line declaration) >> synchronizeEntry path file) `onException` Posix.closeFd file
    Journal <$> newMVar (Just file)

close :: Journal -> IO ()
close (Journal lock) = modifyMVar_ lock (\held -> mapM_ Posix.closeFd held >> pure Nothing)

append :: Journal -> Value -> IO ()
append (Journal lock) entry = withMVar lock (maybe (ioError (userError "Journal is closed")) (\file -> write file (line entry)))

store :: FilePath -> ByteString -> IO ()
store path contents = bracket (exclusive path) Posix.closeFd $ \file -> do
    write file contents
    synchronizeEntry path file

exclusive :: FilePath -> IO Fd
exclusive path = Posix.openFd path Posix.WriteOnly Posix.defaultFileFlags {Posix.append = True, Posix.exclusive = True, Posix.creat = Just 0o644}

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
