{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Batch (Permit, input, authorize, authorizeActivation, permission, qualifications, observe) where

import Control.Monad (unless)
import Data.Aeson (encode, object, (.=))
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Text.Encoding (decodeUtf8)
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Output qualified as Output
import Invar.Infer.Result qualified as Result
import Invar.Qualification qualified as Gate
import Invar.Spec.Invocation qualified as Invocation
import Invar.Spec.Load qualified as Load
import Invar.Spec.Qualification qualified as Qualification

data Permit = Permit ByteString [ByteString] [Call.Permit]

input :: FilePath -> [Call.Call] -> ByteString
input adapter calls = Lazy.toStrict (encode (object ["format" .= Framing.format, "adapter" .= adapter, "calls" .= map (decodeUtf8 . Call.batchInput) calls]))

authorize :: Gate.Registry -> [Call.Call] -> ByteString -> Either Call.Error (Gate.Registry, Permit)
authorize registry calls = authorizeWith Framing.readiness (registry, calls)

authorizeActivation :: Gate.Registry -> [Call.Call] -> ByteString -> Either Call.Error (Gate.Registry, Permit)
authorizeActivation registry calls = authorizeWith Framing.activationReadiness (registry, calls)

authorizeWith :: ([Framing.Frame] -> Either String [ByteString]) -> (Gate.Registry, [Call.Call]) -> ByteString -> Either Call.Error (Gate.Registry, Permit)
authorizeWith readiness (registry, calls) encoded = do
    sources <- first Call.Protocol (Framing.decode encoded >>= readiness)
    unless (length calls == length sources) (Left (Call.Protocol "Batch readiness inventory differs from the declared calls"))
    (updated, permits) <- Call.authorizeBatch registry (zip calls sources)
    pure (updated, Permit encoded sources permits)

permission :: Permit -> ByteString
permission (Permit _ _ permits) = Lazy.toStrict (encode (object ["format" .= Framing.format, "permissions" .= map (decodeUtf8 . Call.permission) permits]))

qualifications :: Permit -> [Maybe Qualification.QualifiedResult]
qualifications (Permit _ _ permits) = map Call.qualified permits

observe :: Permit -> ByteString -> Either Call.Error [(Invocation.Completion, Result.Result, Load.Fact, Maybe Qualification.QualifiedResult)]
observe (Permit prefix sources permits) encoded = do
    unless (prefix `Bytes.isPrefixOf` encoded) (Left (Call.Protocol "Completed batch differs from its authorized prefix"))
    (_, results) <- first Call.Protocol (Framing.decode (Bytes.drop (Bytes.length prefix) encoded) >>= Framing.completion)
    unless (length results == length permits) (Left (Call.Protocol "Batch completion inventory differs from its consumption permits"))
    traverse completed (zip3 permits sources results)
  where
    completed (permit, source, output) = do
        (finished, result) <- Call.observe permit (source <> output)
        first Call.Protocol (Output.rawBehavior output (Result.behaviorBits result))
        pure (finished, result, Call.loadFact permit, Call.qualified permit)
