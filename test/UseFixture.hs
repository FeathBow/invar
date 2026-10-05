{-# LANGUAGE OverloadedStrings #-}

module UseFixture (Trial (..), trials, fixture, repeated, workloadValue, run) where

import Calls (field, request, wire)
import Data.Aeson (Value (..), eitherDecodeStrict, encode, object, toJSON, (.=))
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Text qualified as Text
import Data.Text.Encoding qualified as Text
import Data.Word (Word32)
import Hedgehog
import Invar.Infer qualified as Infer
import Invar.Infer.Invocation qualified as Call
import Invar.Numerical qualified as N
import Invar.Spec.Invocation qualified as Invocation
import Invar.Spec.Load qualified as Load
import Invar.Use qualified as U
import Invar.Use.Decimal qualified as Decimal
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)

data Trial = Trial
    { name :: String
    , prompt :: String
    , seed :: Integer
    , answer :: String
    , before :: String
    , after :: String
    , truncated :: Bool
    }
    deriving (Eq, Show)

trials :: [Trial]
trials =
    [ Trial "a1" "question a" 1 "#### 12" "#### 12" "#### 0" False
    , Trial "a2" "question a" 2 "#### 12" "#### 12" "#### 12" False
    , Trial "a3" "question a" 3 "#### 12" "#### 12" "#### 12" False
    , Trial "b1" "question b" 1 "#### 12" "#### 0" "#### 0" False
    ]

workloadValue :: [Trial] -> Value
workloadValue values = toJSON [object ["tasks" .= map task values, "order" .= inventory, "delivery" .= inventory]]
  where
    inventory = [0 .. length values - 1]
    task value = object ["name" .= name value, "group" .= ("declared-group" :: String), "prompt" .= prompt value, "tokens" .= Infer.tokens request, "temperature" .= Infer.temperature request, "seed" .= seed value, "answer" .= answer value]

fixture :: [Trial] -> PropertyT IO U.BoundRun
fixture = repeated 0

repeated :: Natural -> [Trial] -> PropertyT IO U.BoundRun
repeated count values = do
    document <- evalEither (Workload.decode (Lazy.toStrict (encode (workloadValue values))))
    observations <- traverse paired (zip [0 ..] values)
    evalEither (Decimal.bind document observations)
  where
    paired (index, value) = do
        reference <- run (2 * index) N.Reference value
        candidate <- run (2 * index + 1) N.Candidate value
        repeats <- traverse (\offset -> run (100 + 10 * index + offset) N.Candidate value) [1 .. count]
        pure (U.Case (U.Key 0 (name value)) (N.BoundRun reference candidate) repeats)

run :: Natural -> N.Side -> Trial -> PropertyT IO N.Run
run identity side trial = do
    let requested = request {Infer.prompt = prompt trial, Infer.seed = seed trial}
        binding = Invocation.Binding (Invocation.CallId identity) (Invocation.AttemptId identity) (Invocation.Instance identity)
        bound = object ["call" .= identity, "attempt" .= identity, "instance" .= identity]
        semantic = object ["prompt" .= Infer.prompt requested, "tokens" .= Infer.tokens requested, "temperature" .= Infer.temperature requested, "seed" .= Infer.seed requested]
    planned <- evalEither (Infer.prepare requested)
    called <- evalEither (Call.prepare binding planned)
    envelope <- evalEither (eitherDecodeStrict (Text.encodeUtf8 (Text.pack (Call.input called))))
    let loading = field "load" envelope
        image = Infer.image requested
        materialization = ["tokenizer" .= Infer.tokenizer requested, "base" .= Infer.base requested, "assembly" .= Infer.assembly requested]
        loaded = object (materialization ++ ["stage" .= String "loaded_adapter", "binding" .= bound, "load" .= loading, "image" .= object ["artifact" .= Bytes.unpack (Load.artifact image), "profile" .= Bytes.unpack (Load.profile image)], "requested" .= Infer.artifact requested, "consumed" .= Infer.artifact requested, "model" .= String "test-model", "revision" .= String "test-revision"])
        consumed = object (materialization ++ ["stage" .= String "consumed", "binding" .= bound, "program" .= field "program" envelope, "load" .= loading, "adapter" .= Infer.artifact requested, "request" .= semantic])
        result = object (materialization ++ ["stage" .= String "result", "binding" .= bound, "adapter" .= Infer.artifact requested, "request" .= semantic, "tokens" .= [1, 2, 3 :: Int], "prompt_length" .= Number 1, "behavior" .= [-0.5, -0.25 :: Double], "behavior_bits" .= [0xbf000000, 0xbe800000 :: Word32], "text" .= (case side of N.Reference -> before trial; N.Candidate -> after trial), "truncated" .= truncated trial])
    pure (N.Run planned binding 0 (wire [object ["stage" .= String "load", "cpu_seconds" .= Number 1], loaded, consumed, object ["stage" .= String "inference", "cpu_seconds" .= Number 1], result]) Nothing)
