{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RoleAnnotations #-}

module Invar.Learn.Worker (Mode (..), Worker (..), Call, Hooks (..), Failure (..), Execution, prepare, hooked, input, run, verifyGradients, verifyProbabilities, verifyCheckpoint, report, loaded, plan, staged) where

import Control.Exception (bracket, mask_)
import Data.ByteString qualified as Bytes
import Data.IORef (IORef, modifyIORef', newIORef, readIORef, writeIORef)
import Data.Text qualified as Text
import Data.Text.Encoding (encodeUtf8)
import Invar.Learn qualified as L
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Worker.Internal
import Invar.Process qualified as Process
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as Registry

type role Execution nominal
data Execution scope = Execution (L.Plan scope) P.Result FilePath Registry.Fact

run :: Worker -> Call scope -> IO (Either Failure (Execution scope))
run worker call = bracket (newIORef Registry.empty) (`modifyIORef'` Registry.close) (execute worker call)

execute :: Worker -> Call scope -> IORef Registry.Registry -> IO (Either Failure (Execution scope))
execute worker call@(Call planned binding runtime _ hooks) registry = do
    slot <- newIORef Nothing
    let arguments = [script worker, "--cache=" ++ cache worker, "--checkpoint=" ++ checkpoint worker, "--reference=" ++ reference worker, "--output=" ++ output worker]
        command = Process.Command (executable worker) arguments [] (encodeUtf8 (Text.pack (input call)))
        approve observed = do
            authorized <- authorize (registry, slot) (binding, runtime) observed
            case authorized of
                Left problem -> pure (Left problem)
                Right () -> readIORef slot >>= maybe (pure (Left (P.Unexpected "Update was authorized without a permit"))) (fmap (permission call <$) . announce hooks)
    returned <- Process.conversation command approve (Just (answer hooks slot))
    case returned of
        Right observed -> complete (planned, output worker) slot observed
        Left (Process.Exit status) -> pure (Left (WorkerExit status))
        Left (Process.Rejected problem) -> pure (Left (InvalidOutput problem))
        Left (Process.Protocol problem) -> pure (Left (ProtocolFailure problem))

authorize :: (IORef Registry.Registry, IORef (Maybe P.Permit)) -> (V.Binding, V.Runtime) -> Bytes.ByteString -> IO (Either P.Error ())
authorize (owner, slot) context observed = mask_ $ do
    previous <- readIORef slot
    registry <- readIORef owner
    case previous of
        Just _ -> pure (Left (P.Unexpected "Update already holds a consumption permit"))
        Nothing -> case P.authorize registry context observed of
            Left problem -> pure (Left problem)
            Right (updated, permit) -> do
                writeIORef owner updated
                writeIORef slot (Just permit)
                pure (Right ())

answer :: Hooks -> IORef (Maybe P.Permit) -> Bytes.ByteString -> IO (Either P.Error Bytes.ByteString)
answer hooks slot observed = mask_ $ do
    held <- readIORef slot
    case held of
        Nothing -> pure (Left (P.Unexpected "A learner step was reported before consumption was permitted"))
        Just permit -> case P.respond permit observed of
            Left problem -> pure (Left problem)
            Right (advanced, reply) -> do
                permitted <- consult hooks advanced
                case permitted of
                    Left problem -> pure (Left problem)
                    Right () -> writeIORef slot (Just advanced) >> pure (Right reply)

complete :: (L.Plan scope, FilePath) -> IORef (Maybe P.Permit) -> Bytes.ByteString -> IO (Either Failure (Execution scope))
complete (planned, directory) slot observed = do
    accepted <- readIORef slot
    case accepted of
        Nothing -> pure (Left (ProtocolFailure "Update result has no accepted load and consumption permit"))
        Just permit -> case P.observe permit observed of
            Left problem -> pure (Left (InvalidOutput problem))
            Right result -> fmap (Execution planned result directory (P.loadedFact permit) <$) (verifyArtifacts directory result)

report :: Execution scope -> P.Result
report (Execution _ result _ _) = result

loaded :: Execution scope -> Registry.Fact
loaded (Execution _ _ _ fact) = fact

plan :: Execution scope -> L.Plan scope
plan (Execution planned _ _ _) = planned

staged :: Execution scope -> FilePath
staged (Execution _ _ path _) = path
