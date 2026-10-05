{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RoleAnnotations #-}

module Invar.Process.Resident (Resident, Handshake (..), Transaction (..), withResident, exchange, hosted, initial) where

import Control.Concurrent.MVar (MVar, newMVar, withMVar)
import Control.Exception (mask, onException)
import Data.ByteString (ByteString)
import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import Data.Kind (Type)
import Invar.Process.Internal qualified as Process
import Invar.Resident qualified as Boundary
import Invar.Resident.Owner qualified as Owner

data Handshake problem value = Handshake {message :: ByteString, admit :: ByteString -> IO (Either problem value)}
data Transaction problem = Transaction {invocation :: Process.Exchange problem, retirement :: ByteString -> IO (Either problem (Handshake problem Owner.Released))}
data State = Idle Owner.State | Failed | Closed

type role Resident nominal
data Resident (scope :: Type) = Resident Process.Pipes (MVar ()) (IORef State)

withResident :: Process.Launch -> Boundary.Owner -> (forall scope. Resident scope -> IO (Either problem ())) -> (forall scope. Resident scope -> IO (Either (Process.Failure problem) value)) -> IO (Either (Process.Failure problem) value)
withResident launch selected ready action = Process.withLaunch launch $ \handles -> do
    resident <- Resident handles <$> newMVar () <*> newIORef (Idle (Owner.start selected))
    returned <- action resident `onException` poison resident (Process.Protocol "Resident owner was interrupted")
    case returned of
        Left problem -> poison resident problem
        Right value -> fmap (value <$) (close resident (ready resident))

exchange :: Resident scope -> Transaction problem -> IO (Either (Process.Failure problem) (ByteString, ByteString))
exchange resident transaction = serialized resident $ \current -> do
    let session = channel resident
        selected = invocation transaction
        active = session {Process.review = Process.permission selected}
    Process.send (Process.pipes session) (Process.message selected)
    returned <- Process.response active selected ([], False)
    case returned of
        Left problem -> pure (Left problem)
        Right output -> do
            release <- retirement transaction output
            case release of
                Left problem -> Process.reject session (Process.Rejected problem)
                Right expected -> do
                    acknowledged <- handshake session expected
                    case acknowledged of
                        Left problem -> pure (Left problem)
                        Right (encoded, released) -> case Owner.release current released of
                            Left problem -> Process.reject session (Process.Protocol problem)
                            Right following -> pure (Right ((output, encoded), Idle following))

hosted :: Resident scope -> (Process.Pipes -> Owner.State -> IO (Either (Process.Failure problem) (value, Owner.State))) -> IO (Either (Process.Failure problem) value)
hosted resident@(Resident pipes _ _) action = serialized resident (fmap (fmap (fmap Idle)) . action pipes)

initial :: Resident scope -> IO Bool
initial (Resident _ _ state) = do
    current <- readIORef state
    case current of
        Idle physical -> pure (Owner.initial physical)
        _ -> ioError (userError "Resident group count requested outside an active owner")

channel :: Resident scope -> Process.Session problem
channel (Resident pipes _ _) = Process.Session pipes (const (pure (Right "")))

handshake :: Process.Session problem -> Handshake problem value -> IO (Either (Process.Failure problem) (ByteString, value))
handshake session expected = do
    Process.send (Process.pipes session) (message expected)
    returned <- Process.line session
    case returned of
        Left problem -> pure (Left problem)
        Right observed -> do
            accepted <- admit expected observed
            case accepted of
                Left problem -> Process.reject session (Process.Rejected problem)
                Right value -> pure (Right (observed, value))

serialized :: Resident scope -> (Owner.State -> IO (Either (Process.Failure problem) (value, State))) -> IO (Either (Process.Failure problem) value)
serialized resident@(Resident _ gate state) action = withMVar gate $ \() -> mask $ \restore -> do
    current <- readIORef state
    case current of
        Failed -> pure (Left (Process.Protocol "Resident worker previously failed"))
        Closed -> pure (Left (Process.Protocol "Resident worker is already closed"))
        Idle physical -> do
            returned <- restore (action physical) `onException` poison resident (Process.Protocol "Resident exchange was interrupted")
            case returned of
                Left problem -> writeIORef state Failed >> pure (Left problem)
                Right (value, following) -> writeIORef state following >> pure (Right value)

close :: Resident scope -> IO (Either problem ()) -> IO (Either (Process.Failure problem) ())
close resident ready = serialized resident $ \current -> do
    let session = channel resident
    checked <- ready
    case checked of
        Left problem -> Process.reject session (Process.Rejected problem)
        Right () -> do
            received <- handshake session (Handshake (Boundary.close (Owner.owner current)) (const (pure (Right ()))))
            case received of
                Left problem -> pure (Left problem)
                Right (encoded, ()) -> case Owner.close current encoded of
                    Left problem -> Process.reject session (Process.Protocol problem)
                    Right () -> do
                        finished <- Process.finish session
                        case finished of
                            Left problem -> pure (Left problem)
                            Right () -> pure (Right ((), Closed))

poison :: Resident scope -> Process.Failure problem -> IO (Either (Process.Failure problem) value)
poison resident@(Resident _ _ state) problem = do
    writeIORef state Failed
    Process.reject (channel resident) problem
