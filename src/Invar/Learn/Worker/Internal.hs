{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RoleAnnotations #-}

module Invar.Learn.Worker.Internal (Mode (..), Worker (..), Call (..), Failure (..), prepare, input, permission, verifyArtifacts, verifyGradients, verifyProbabilities, verifyCheckpoint) where

import Control.Exception (bracket)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (encode, object, (.=))
import Data.Bifunctor (first)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8)
import Invar.Artifact qualified as Artifact
import Invar.Infer.Wire qualified as Binding
import Invar.Learn qualified as L
import Invar.Learn.Probability qualified as Probability
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Wire qualified as Wire
import Invar.Load qualified as Load
import Invar.Policy qualified as Policy
import Invar.Spec.Invocation qualified as V
import System.Exit (ExitCode)
import System.FilePath ((</>))
import System.IO (hClose)

data Mode = Process | Resident | Shared deriving (Eq, Show)
data Worker = Worker {executable :: FilePath, script :: FilePath, cache :: FilePath, checkpoint :: FilePath, reference :: FilePath, output :: FilePath}

type role Call nominal
data Call scope = Call (L.Plan scope) V.Binding V.Runtime String

data Failure = Preparation L.Error | Lowering Wire.Error | Loading Load.Error | WorkerExit ExitCode | InvalidOutput P.Error | GradientMismatch String String | ProbabilityMismatch String String | InvalidProbability String | PolicyMismatch String String | LearnerMismatch String String | ProtocolFailure String
    deriving (Eq, Show)

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

permission :: Call scope -> Bytes.ByteString
permission (Call planned binding _ _) = Lazy.toStrict (encode (Binding.invocationValue binding (L.program planned)))

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
