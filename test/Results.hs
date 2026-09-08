{-# LANGUAGE OverloadedStrings #-}

module Results (results) where

import Control.Monad (forM_)
import Data.Aeson (Value (..), encode, object, toJSON, (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Map
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Word (Word32)
import Hedgehog
import Invar.Infer qualified as I
import Invar.Infer.Result qualified as R

results :: Group
results = Group "Inference report correspondence" [("matching reports preserve actual token observations", once matching), ("all checked request fields bind the result", once bindings), ("malformed token observations cannot complete", once shapes), ("missing repeated and reordered reports are rejected", once protocol)]
  where
    once = withTests 1 . property

identity :: String
identity = replicate 64 'a'

planned :: PropertyT IO I.Plan
planned = evalEither (I.prepare I.Request {I.artifact = identity, I.tokenizer = replicate 64 'c', I.base = replicate 64 'e', I.assembly = replicate 64 'f', I.prompt = "example", I.tokens = 2, I.temperature = 0.8, I.seed = 17})

load :: Value
load = object ["stage" .= String "loaded_adapter", "requested" .= identity, "consumed" .= identity, "tokenizer" .= replicate 64 'c', "base" .= replicate 64 'e', "assembly" .= replicate 64 'f']

request :: Value
request = object ["prompt" .= String "example", "tokens" .= Number 2, "temperature" .= Number 0.8, "seed" .= Number 17]

result :: Value
result = object ["stage" .= String "result", "adapter" .= identity, "tokenizer" .= replicate 64 'c', "base" .= replicate 64 'e', "assembly" .= replicate 64 'f', "request" .= request, "tokens" .= [1, 2, 3 :: Int], "prompt_length" .= Number 1, "behavior" .= [-0.5, -0.25 :: Double], "behavior_bits" .= [0xbf000000, 0xbe800000 :: Word32], "truncated" .= True, "text" .= String "reported response"]

wire :: [Value] -> ByteString
wire = Bytes.unlines . map (Lazy.toStrict . encode)

change :: (Key, Value) -> Value -> Value
change (key, value) (Object fields) = Object (Map.insert key value fields)
change _ _ = error "Report fixture must be an object"

matching :: PropertyT IO ()
matching = do
    plan <- planned
    observed <- evalEither (R.observe plan (wire [load, result]))
    R.tokens observed === [1, 2, 3]
    R.behavior observed === [-0.5, -0.25]
    R.behaviorBits observed === [0xbf000000, 0xbe800000]
    R.promptLength observed === 1
    R.truncated observed === True
    R.consumed observed === I.requested plan
    R.response observed === "reported response"
    let zero = change ("behavior", toJSON [Number 0, Number (-0.25)]) result
    negative <- evalEither (R.observe plan (wire [load, change ("behavior_bits", toJSON [0x80000000, 0xbe800000 :: Word32]) zero]))
    positive <- evalEither (R.observe plan (wire [load, change ("behavior_bits", toJSON [0, 0xbe800000 :: Word32]) zero]))
    assert (negative /= positive)
    R.behaviorBits negative === [0x80000000, 0xbe800000]

bindings :: PropertyT IO ()
bindings = do
    plan <- planned
    forM_ [("prompt", String "other"), ("tokens", Number 3), ("temperature", Number 1), ("seed", Number 18)] $ \field ->
        mismatch (R.observe plan (wire [load, change ("request", change field request) result]))
    mismatch (R.observe plan (wire [load, change ("adapter", String "other") result]))
    forM_ ["tokenizer", "base", "assembly"] $ \field ->
        mismatch (R.observe plan (wire [load, change (field, toJSON (replicate 64 '0')) result]))
    forM_ ["requested", "consumed", "tokenizer", "base", "assembly"] $ \field ->
        mismatch (R.observe plan (wire [change (field, String "other") load, result]))

shapes :: PropertyT IO ()
shapes = do
    plan <- planned
    forM_ [("prompt_length", Number 0), ("prompt_length", Number 3), ("behavior", toJSON [Number (-0.5)]), ("behavior", toJSON [Number 0.5, Number (-0.25)]), ("tokens", toJSON [Number 1, Number 2, Number 3, Number 4])] $ \field ->
        mismatch (R.observe plan (wire [load, change field result]))
    let short = change ("tokens", toJSON [Number 1, Number 2]) (change ("behavior", toJSON [Number (-0.5)]) result)
    mismatch (R.observe plan (wire [load, short]))
    mismatch (R.observe plan (wire [load, change ("behavior_bits", toJSON [0, 0xbe800000 :: Word32]) result]))
    mismatch (R.observe plan (wire [load, change ("behavior_bits", toJSON [0x7fc00000, 0xbe800000 :: Word32]) result]))

protocol :: PropertyT IO ()
protocol = do
    plan <- planned
    forM_ [[], [result], [load], [load, load, result], [load, result, result], [object ["stage" .= String "unknown"]]] $ \events ->
        case R.observe plan (wire events) of
            Left (R.Unexpected _) -> success
            unexpected -> annotateShow unexpected >> failure
    forM_ ("not json" : [wire [load, change (field, Null) result] | field <- ["request", "text", "base", "assembly"]]) $ \encoded ->
        case R.observe plan encoded of
            Left (R.Malformed _) -> success
            unexpected -> annotateShow unexpected >> failure

mismatch :: Either R.Error R.Result -> PropertyT IO ()
mismatch outcome = case outcome of
    Left (R.Mismatch _) -> success
    unexpected -> annotateShow unexpected >> failure
