module Invar.Transcript (Transcript (..), Outcome (..), Output (..), standard, echoing, live) where

import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import System.Exit (ExitCode)
import System.IO (hFlush, stdout)

data Transcript = Transcript {record :: ByteString -> IO (), partial :: ByteString -> IO (), finished :: Outcome -> IO ()}

data Outcome = Unlaunched String | Exited ExitCode Output | Stopped Output
    deriving (Eq, Show)

data Output = Complete | Cut
    deriving (Eq, Show)

standard :: Transcript
standard = echoing live

echoing :: (ByteString -> IO ()) -> Transcript
echoing writer = Transcript writer writer (const (pure ()))

live :: ByteString -> IO ()
live line = Bytes.hPut stdout (Bytes.snoc line '\n') >> hFlush stdout
