module Invar.Measurement.Source (Source (..), Snapshot (..), snapshot, invalid, checked) where

import Crypto.Hash.SHA256 qualified as SHA256
import Data.ByteString (ByteString)
import Invar.Artifact qualified as Artifact

data Source = Source {readBytes :: FilePath -> IO ByteString, resolvePath :: FilePath -> IO FilePath}
data Snapshot = Snapshot {encoded :: ByteString, digest :: String}

snapshot :: Source -> FilePath -> IO Snapshot
snapshot source path = do
    bytes <- readBytes source path
    pure (Snapshot bytes (Artifact.hex (SHA256.hash bytes)))

invalid :: String -> IO value
invalid = ioError . userError

checked :: Either String value -> IO value
checked = either invalid pure
