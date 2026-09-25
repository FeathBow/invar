{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Batch (Permit, Reference (..), input, authorize, authorizeActivation, permission, observe) where

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
import Invar.Spec.Invocation qualified as Invocation
import Invar.Spec.Load qualified as Load

data Permit = Permit ByteString [ByteString] [Call.Permit]

data Reference = Reference {location :: FilePath, identity :: String}
    deriving (Eq, Show)

input :: FilePath -> Maybe Reference -> [Call.Call] -> ByteString
input adapter reference calls = Lazy.toStrict (encode (object ["format" .= Framing.format, "adapter" .= adapter, "reference" .= fmap declared reference, "calls" .= map (decodeUtf8 . Call.batchInput) calls]))
  where
    declared selected = object ["adapter" .= location selected, "digest" .= identity selected]

authorize :: Load.Registry -> [Call.Call] -> ByteString -> Either Call.Error (Load.Registry, Permit)
authorize registry calls = authorizeWith Framing.readiness (registry, calls)

authorizeActivation :: Load.Registry -> [Call.Call] -> ByteString -> Either Call.Error (Load.Registry, Permit)
authorizeActivation registry calls = authorizeWith Framing.activationReadiness (registry, calls)

authorizeWith :: ([Framing.Frame] -> Either String [ByteString]) -> (Load.Registry, [Call.Call]) -> ByteString -> Either Call.Error (Load.Registry, Permit)
authorizeWith readiness (registry, calls) encoded = do
    sources <- first Call.Protocol (Framing.decode encoded >>= readiness)
    unless (length calls == length sources) (Left (Call.Protocol "Batch readiness inventory differs from the declared calls"))
    (updated, permits) <- Call.authorizeBatch registry (zip calls sources)
    pure (updated, Permit encoded sources permits)

permission :: Permit -> ByteString
permission (Permit _ _ permits) = Lazy.toStrict (encode (object ["format" .= Framing.format, "permissions" .= map (decodeUtf8 . Call.permission) permits]))

observe :: Permit -> ByteString -> Either Call.Error [(Invocation.Completion, Result.Result, Load.Fact)]
observe (Permit prefix sources permits) encoded = do
    unless (prefix `Bytes.isPrefixOf` encoded) (Left (Call.Protocol "Completed batch differs from its authorized prefix"))
    (_, results) <- first Call.Protocol (Framing.decode (Bytes.drop (Bytes.length prefix) encoded) >>= Framing.completion)
    unless (length results == length permits) (Left (Call.Protocol "Batch completion inventory differs from its consumption permits"))
    traverse completed (zip3 permits sources results)
  where
    completed (permit, source, output) = do
        (finished, result) <- Call.observe permit (source <> output)
        first Call.Protocol (Output.rawBehavior output (Result.behaviorBits result))
        pure (finished, result, Call.loadFact permit)
