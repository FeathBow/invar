{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RoleAnnotations #-}

module Invar.Process.Resident (Resident, Handshake (..), Transaction (..), withResident, exchange, groups) where

import Control.Concurrent.MVar (MVar, newMVar, withMVar)
import Control.Exception (mask, onException)
import Data.ByteString (ByteString)
import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import Data.Kind (Type)
import Invar.Process.Internal qualified as Process
import Numeric.Natural (Natural)

data Handshake problem = Handshake {message :: ByteString, admit :: ByteString -> IO (Either problem ())}
data Transaction problem = Transaction {invocation :: Process.Exchange problem, retirement :: ByteString -> IO (Either problem (Handshake problem))}
data State = Idle Natural | Failed | Closed

type role Resident nominal
data Resident (scope :: Type) = Resident Process.Pipes (ByteString -> IO ()) (MVar ()) (IORef State)

withResident :: Process.Launch -> (forall scope. Resident scope -> Handshake problem) -> (forall scope. Resident scope -> IO (Either (Process.Failure problem) value)) -> IO (Either (Process.Failure problem) value)
withResident launch closing action = Process.withLaunch launch $ \handles -> do
    resident <- Resident handles (Process.echo launch) <$> newMVar () <*> newIORef (Idle 0)
    returned <- action resident `onException` poison resident (Process.Protocol "Resident owner was interrupted")
    case returned of
        Left problem -> poison resident problem
        Right value -> fmap (value <$) (close resident (closing resident))

exchange :: Resident scope -> Transaction problem -> IO (Either (Process.Failure problem) (ByteString, ByteString))
exchange resident@(Resident _ _ _ state) transaction = serialized resident $ do
    count <- groups resident
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
                        Right encoded -> writeIORef state (Idle (count + 1)) >> pure (Right (output, encoded))

channel :: Resident scope -> Process.Session problem
channel (Resident pipes emit _ _) = Process.Session pipes (const (pure (Right ""))) emit

groups :: Resident scope -> IO Natural
groups (Resident _ _ _ state) = do
    current <- readIORef state
    case current of
        Idle count -> pure count
        _ -> ioError (userError "Resident group count requested outside an active owner")

handshake :: Process.Session problem -> Handshake problem -> IO (Either (Process.Failure problem) ByteString)
handshake session expected = do
    Process.send (Process.pipes session) (message expected)
    returned <- Process.line session
    case returned of
        Left problem -> pure (Left problem)
        Right observed -> do
            accepted <- admit expected observed
            case accepted of
                Left problem -> Process.reject session (Process.Rejected problem)
                Right () -> pure (Right observed)

serialized :: Resident scope -> IO (Either (Process.Failure problem) value) -> IO (Either (Process.Failure problem) value)
serialized resident@(Resident _ _ gate state) action = withMVar gate $ \() -> mask $ \restore -> do
    current <- readIORef state
    case current of
        Failed -> pure (Left (Process.Protocol "Resident worker previously failed"))
        Closed -> pure (Left (Process.Protocol "Resident worker is already closed"))
        Idle _ -> do
            returned <- restore action `onException` poison resident (Process.Protocol "Resident exchange was interrupted")
            case returned of
                Left _ -> writeIORef state Failed
                Right _ -> pure ()
            pure returned

close :: Resident scope -> Handshake problem -> IO (Either (Process.Failure problem) ())
close resident@(Resident _ _ _ state) closing = serialized resident $ do
    let session = channel resident
    received <- handshake session closing
    case received of
        Left problem -> pure (Left problem)
        Right _ -> do
            finished <- Process.finish session
            case finished of
                Left problem -> pure (Left problem)
                Right () -> writeIORef state Closed >> pure (Right ())

poison :: Resident scope -> Process.Failure problem -> IO (Either (Process.Failure problem) value)
poison resident@(Resident _ _ _ state) problem = do
    writeIORef state Failed
    Process.reject (channel resident) problem
