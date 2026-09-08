{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RoleAnnotations #-}

module Invar.Learn.Worker (Worker (..), Call, Failure (..), Execution, prepare, input, run, verifyGradients, verifyProbabilities, verifyCheckpoint, report, loaded, plan, staged) where

import Control.Exception (bracket, mask_)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (encode, object, (.=))
import Data.Bifunctor (first)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.IORef (IORef, modifyIORef', newIORef, readIORef, writeIORef)
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8, encodeUtf8)
import Invar.Artifact qualified as Artifact
import Invar.Infer.Wire qualified as Binding
import Invar.Learn qualified as L
import Invar.Learn.Probability qualified as Probability
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Wire qualified as Wire
import Invar.Load qualified as Load
import Invar.Policy qualified as Policy
import Invar.Process qualified as Process
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as Registry
import System.Exit (ExitCode)
import System.FilePath ((</>))
import System.IO (hClose)

data Worker = Worker {executable :: FilePath, script :: FilePath, cache :: FilePath, checkpoint :: FilePath, reference :: FilePath, output :: FilePath}

type role Call nominal
data Call scope = Call (L.Plan scope) V.Binding V.Runtime String

data Failure = Preparation L.Error | Lowering Wire.Error | Loading Load.Error | WorkerExit ExitCode | InvalidOutput P.Error | GradientMismatch String String | ProbabilityMismatch String String | InvalidProbability String | PolicyMismatch String String | LearnerMismatch String String | ProtocolFailure String
    deriving (Eq, Show)

type role Execution nominal
data Execution scope = Execution (L.Plan scope) P.Result FilePath Registry.Fact

prepare :: V.Binding -> L.Plan scope -> Either Failure (Call scope)
prepare binding planned = do
    runtime <- first Preparation (L.invocation binding planned)
    request <- first Lowering (Wire.lower (L.emission planned))
    image <- first Lowering (Wire.image (L.emission planned))
    loading <- first Loading (Load.prepare binding image)
    let invocation = Binding.invocationValue binding (L.program planned)
        encoded = encode (object ["invocation" .= invocation, "request" .= request, "load" .= Binding.invocationValue binding (Load.program loading)])
    pure (Call planned binding runtime (Text.unpack (decodeUtf8 (Lazy.toStrict encoded))))

input :: Call scope -> String
input (Call _ _ _ encoded) = encoded

run :: Worker -> Call scope -> IO (Either Failure (Execution scope))
run worker call = bracket (newIORef Registry.empty) (`modifyIORef'` Registry.close) (execute worker call)

execute :: Worker -> Call scope -> IORef Registry.Registry -> IO (Either Failure (Execution scope))
execute worker call@(Call planned binding runtime _) registry = do
    slot <- newIORef Nothing
    let arguments = [script worker, "--cache=" ++ cache worker, "--checkpoint=" ++ checkpoint worker, "--reference=" ++ reference worker, "--output=" ++ output worker]
        command = Process.Command (executable worker) arguments [] (encodeUtf8 (Text.pack (input call)))
        permission = Lazy.toStrict (encode (Binding.invocationValue binding (L.program planned)))
        approve observed = fmap (permission <$) (authorize (registry, slot) (binding, runtime) observed)
    returned <- Process.run command approve
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

complete :: (L.Plan scope, FilePath) -> IORef (Maybe P.Permit) -> Bytes.ByteString -> IO (Either Failure (Execution scope))
complete (planned, directory) slot observed = do
    accepted <- readIORef slot
    case accepted of
        Nothing -> pure (Left (ProtocolFailure "Update result has no accepted load and consumption permit"))
        Just permit -> case P.observe permit observed of
            Left problem -> pure (Left (InvalidOutput problem))
            Right result -> fmap (Execution planned result directory (P.loadedFact permit) <$) (verifyArtifacts directory result)

verifyGradients :: FilePath -> P.Result -> IO (Either Failure ())
verifyGradients directory result = do
    actual <- Artifact.identity "Gradient observation" (directory </> "gradients.safetensors")
    let expected = P.gradients result
    pure $ if actual == expected then Right () else Left (GradientMismatch expected actual)

verifyCheckpoint :: FilePath -> P.Result -> IO (Either Failure ())
verifyCheckpoint directory result = do
    actual <- Policy.identity (directory </> "adapter.safetensors")
    if actual /= P.adapter result
        then pure (Left (PolicyMismatch (P.adapter result) actual))
        else do
            learner <- Artifact.identity "Learner checkpoint" (directory </> "learner.pt")
            pure $ if learner == P.learner result then Right () else Left (LearnerMismatch (P.learner result) learner)

verifyProbabilities :: FilePath -> P.Result -> IO (Either Failure ())
verifyProbabilities directory result = do
    encoded <- bracket (Artifact.open "Probability observation" (directory </> "probabilities.json")) hClose Bytes.hGetContents
    let actual = Artifact.hex (SHA256.hash encoded)
        expected = P.probabilities result
    pure $ if actual /= expected then Left (ProbabilityMismatch expected actual) else first InvalidProbability (Probability.validate result encoded)

verifyArtifacts :: FilePath -> P.Result -> IO (Either Failure ())
verifyArtifacts directory result = checks [verifyGradients, verifyProbabilities, verifyCheckpoint]
  where
    checks [] = pure (Right ())
    checks (check : remaining) = do
        actual <- check directory result
        case actual of
            Left problem -> pure (Left problem)
            Right () -> checks remaining

report :: Execution scope -> P.Result
report (Execution _ result _ _) = result

loaded :: Execution scope -> Registry.Fact
loaded (Execution _ _ _ fact) = fact

plan :: Execution scope -> L.Plan scope
plan (Execution planned _ _ _) = planned

staged :: Execution scope -> FilePath
staged (Execution _ _ path _) = path
