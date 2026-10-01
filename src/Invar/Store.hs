{-# LANGUAGE OverloadedStrings #-}

module Invar.Store (
    Location (..),
    Receipt,
    location,
    Method (..),
    method,
    methodName,
    Phase (..),
    Failure (..),
    publish,
    publishCheckpoint,
    synchronizeFile,
    synchronizeDirectory,
) where

import Control.Exception (Exception, IOException, bracket, mask_, throwIO)
import Control.Monad (unless, when)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Foreign.C.Error (throwErrnoIfMinus1_)
import Foreign.C.String (CString)
import Foreign.C.Types (CInt (..))
import System.FilePath (dropTrailingPathSeparator, takeDirectory)
import System.IO.Error (catchIOError)
import System.Posix.Files (getFdStatus, isRegularFile)
import System.Posix.IO qualified as Posix
import System.Posix.IO.ByteString qualified as Raw
import System.Posix.Types (Fd (..))

data Location = Location
    { directory :: FilePath
    , staging :: ByteString
    , destination :: ByteString
    }
    deriving (Eq, Show)

data Method = RenameExclusive | LinkImmutable
    deriving (Eq, Show)

methodName :: Method -> String
methodName RenameExclusive = "rename"
methodName LinkImmutable = "reference"

data Receipt = Receipt Method Location
    deriving (Eq, Show)

data Parent = Parent {entries :: Fd, namespace :: Fd}

location :: Receipt -> Location
location (Receipt _ value) = value

method :: Receipt -> Method
method (Receipt chosen _) = chosen

data Phase = Validate | Prepare | Rename | Link | Synchronize | Close
    deriving (Eq, Show)

data Failure = Failure Phase IOException
    deriving (Show)

instance Exception Failure

publish :: Location -> IO Receipt
publish target = withParent target $ \parent ->
    withDescriptor (openEntry (entries parent) (staging target) fileFlags) (commit target parent)

publishCheckpoint :: Method -> Location -> IO Receipt
publishCheckpoint chosen target = withParent target $ \parent ->
    withDescriptor (openEntry (entries parent) (staging target) directoryFlags) $ \staged ->
        commitCheckpoint chosen target (parent, staged)

withParent :: Location -> (Parent -> IO value) -> IO value
withParent target operation = mask_ $ do
    validateLocation target
    let containing = directory target
        ancestor = takeDirectory (dropTrailingPathSeparator containing)
    withDescriptor (Posix.openFd containing Posix.ReadOnly directoryFlags) $ \parent ->
        withDescriptor (Posix.openFd ancestor Posix.ReadOnly directoryFlags) $ \outer ->
            operation (Parent parent outer)

withDescriptor :: IO Fd -> (Fd -> IO value) -> IO value
withDescriptor acquire = bracket (at Prepare acquire) (at Close . Posix.closeFd)

fileFlags, directoryFlags :: Posix.OpenFileFlags
fileFlags = Posix.defaultFileFlags {Posix.nofollow = True, Posix.cloexec = True, Posix.nonBlock = True}
directoryFlags = fileFlags {Posix.directory = True}

openEntry :: Fd -> ByteString -> Posix.OpenFileFlags -> IO Fd
openEntry parent name = Raw.openFdAt (Just parent) name Posix.ReadOnly

commit :: Location -> Parent -> Fd -> IO Receipt
commit target parent file = do
    at Prepare (prepareFile file)
    at Rename (renameExclusive (entries parent) target)
    at Synchronize (synchronizeParents parent file)
    pure (Receipt RenameExclusive target)

commitCheckpoint :: Method -> Location -> (Parent, Fd) -> IO Receipt
commitCheckpoint chosen target (parent, staged) = withCheckpoint staged $ \(adapter, learner, policy) -> do
    at Prepare $ do
        prepareFile adapter
        prepareFile learner
        prepareFile policy
        synchronizeDirectory staged learner
    case chosen of
        RenameExclusive -> at Rename (renameExclusive (entries parent) target)
        LinkImmutable -> at Link (linkImmutable (entries parent) target)
    at Synchronize (synchronizeParents parent learner)
    pure (Receipt chosen target)

withCheckpoint :: Fd -> ((Fd, Fd, Fd) -> IO value) -> IO value
withCheckpoint staged operation =
    withDescriptor (openEntry staged "adapter.safetensors" fileFlags) $ \adapter ->
        withDescriptor (openEntry staged "learner.pt" fileFlags) $ \learner ->
            withDescriptor (openEntry staged "policy.json" fileFlags) $ \policy ->
                operation (adapter, learner, policy)

prepareFile :: Fd -> IO ()
prepareFile file = do
    status <- getFdStatus file
    unless (isRegularFile status) (ioError (userError "Staged artifact is not a regular file"))
    synchronizeFile file

validateLocation :: Location -> IO ()
validateLocation target = at Validate $ do
    when ('\0' `elem` directory target) (ioError (userError "Directory contains NUL"))
    unless (validName (staging target) && validName (destination target)) $
        ioError (userError "Artifact names must be single nonempty path components")
    when (staging target == destination target) (ioError (userError "Staging and destination names must differ"))
  where
    validName name =
        not (Bytes.null name)
            && name /= Bytes.singleton dot
            && name /= Bytes.pack [dot, dot]
            && not (Bytes.any (\byte -> byte == nul || byte == slash) name)
    nul = 0
    dot = 46
    slash = 47

at :: Phase -> IO value -> IO value
at phase operation = catchIOError operation (throwIO . Failure phase)

renameExclusive :: Fd -> Location -> IO ()
renameExclusive (Fd parent) target =
    Bytes.useAsCString (staging target) $ \source ->
        Bytes.useAsCString (destination target) $ \result ->
            throwErrnoIfMinus1_ "publish rename" (renameFile parent source result)

linkImmutable :: Fd -> Location -> IO ()
linkImmutable (Fd parent) target =
    Bytes.useAsCString (staging target) $ \source ->
        Bytes.useAsCString (destination target) $ \result ->
            throwErrnoIfMinus1_ "publish reference" (linkDirectory parent source result)

synchronizeFile :: Fd -> IO ()
synchronizeFile (Fd file) = throwErrnoIfMinus1_ "publish file sync" (syncFile file)

synchronizeDirectory :: Fd -> Fd -> IO ()
synchronizeDirectory (Fd parent) (Fd file) =
    throwErrnoIfMinus1_ "publish directory sync" (syncDirectory parent file)

synchronizeParents :: Parent -> Fd -> IO ()
synchronizeParents parent file = do
    synchronizeDirectory (entries parent) file
    synchronizeDirectory (namespace parent) file

foreign import ccall safe "invar_rename" renameFile :: CInt -> CString -> CString -> IO CInt
foreign import ccall safe "invar_reference" linkDirectory :: CInt -> CString -> CString -> IO CInt
foreign import ccall safe "invar_sync_file" syncFile :: CInt -> IO CInt
foreign import ccall safe "invar_sync_directory" syncDirectory :: CInt -> CInt -> IO CInt
