module Invar.Transcript (Transcript (..), Outcome (..), standard, echoing, live) where

import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import System.Exit (ExitCode)
import System.IO (hFlush, stdout)

data Transcript = Transcript {record :: ByteString -> IO (), finished :: Outcome -> IO ()}

data Outcome = Unlaunched String | Exited ExitCode | Stopped
    deriving (Eq, Show)

standard :: Transcript
standard = echoing live

echoing :: (ByteString -> IO ()) -> Transcript
echoing writer = Transcript writer (const (pure ()))

live :: ByteString -> IO ()
live line = Bytes.hPut stdout (Bytes.snoc line '\n') >> hFlush stdout
