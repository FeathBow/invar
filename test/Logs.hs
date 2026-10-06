{-# LANGUAGE OverloadedStrings #-}

module Logs (single, admitted) where

import Calls (field, wire)
import Data.Aeson (Value (..), eitherDecodeStrict, object, (.=))
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text qualified as Text
import Data.Text.Encoding qualified as Text
import Data.Word (Word32)
import Hedgehog
import Invar.Infer qualified as Infer
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Session qualified as Session
import Invar.Infer.Trajectory (Trajectory)
import Invar.Spec.Invocation qualified as Invocation
import Invar.Spec.Load qualified as Load
import System.Exit (ExitCode (..))

single :: Infer.Plan -> Invocation.Binding -> (String, Bool) -> PropertyT IO ByteString
single planned binding (text, limited) = do
    called <- evalEither (Call.prepare binding planned)
    envelope <- evalEither (eitherDecodeStrict (Text.encodeUtf8 (Text.pack (Call.input called))))
    let requested = Infer.requested planned
        Invocation.Binding (Invocation.CallId call) (Invocation.AttemptId attempt) (Invocation.Instance instanceId) = binding
        bound = object ["call" .= call, "attempt" .= attempt, "instance" .= instanceId]
        semantic = object ["prompt" .= Infer.prompt requested, "tokens" .= Infer.tokens requested, "temperature" .= Infer.temperature requested, "seed" .= Infer.seed requested]
        loading = field "load" envelope
        image = Infer.image requested
        materialization = ["tokenizer" .= Infer.tokenizer requested, "base" .= Infer.base requested, "assembly" .= Infer.assembly requested]
        loaded = object (materialization ++ ["stage" .= String "loaded_adapter", "binding" .= bound, "load" .= loading, "image" .= object ["artifact" .= Bytes.unpack (Load.artifact image), "profile" .= Bytes.unpack (Load.profile image)], "requested" .= Infer.artifact requested, "consumed" .= Infer.artifact requested, "model" .= String "test-model", "revision" .= String "test-revision"])
        consumed = object (materialization ++ ["stage" .= String "consumed", "binding" .= bound, "program" .= field "program" envelope, "load" .= loading, "adapter" .= Infer.artifact requested, "request" .= semantic])
        result = object (materialization ++ ["stage" .= String "result", "binding" .= bound, "adapter" .= Infer.artifact requested, "request" .= semantic, "tokens" .= [1, 2, 3 :: Int], "prompt_length" .= Number 1, "behavior" .= [-0.5, -0.25 :: Double], "behavior_bits" .= [0xbf000000, 0xbe800000 :: Word32], "text" .= text, "truncated" .= limited])
    pure (wire [object ["stage" .= String "load", "cpu_seconds" .= Number 1], loaded, consumed, object ["stage" .= String "inference", "cpu_seconds" .= Number 1], result])

admitted :: Infer.Plan -> Invocation.Binding -> (String, Bool) -> PropertyT IO Trajectory
admitted planned binding output = do
    encoded <- single planned binding output
    called <- evalEither (Call.prepare binding planned)
    logged <- evalEither (either (Left . show) Right (Replay.standalone Session.Single (Session.Declaration [called] Nothing) ExitSuccess encoded))
    case logged of
        [one] -> pure (Replay.trajectory one)
        _ -> failure
