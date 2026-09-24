{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Observation (Report, admit, admitFrames, admitGroup, result, binding, logDigest, policyDescription, describe) where

import Control.Monad (unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Object, Value (..), object, withObject, (.:), (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text (Text)
import Data.Text qualified as Text
import Invar.Artifact qualified as Artifact
import Invar.Infer qualified as Infer
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Invocation qualified as Invocation
import Invar.Infer.Output qualified as Output
import Invar.Infer.Result qualified as Result
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Policy qualified as Policy
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as Load

data Report = Report String V.Binding Result.Result Object
    deriving (Eq, Show)

admit :: Infer.Plan -> V.Binding -> ByteString -> Either String Report
admit planned bound encoded = do
    unless ("\n" `Bytes.isSuffixOf` encoded) (Left "Incomplete final inference observation line")
    Framing.decode encoded >>= admitWith planned bound encoded

admitFrames :: Infer.Plan -> V.Binding -> [Framing.Frame] -> Either String Report
admitFrames planned bound records = admitWith planned bound (Framing.encode records) records

admitWith :: Infer.Plan -> V.Binding -> ByteString -> [Framing.Frame] -> Either String Report
admitWith planned bound encoded records =
    case break Framing.grouped records of
        (_, []) -> do
            observed <- trace [(Framing.raw frame, Framing.fields frame) | frame <- records]
            check planned bound (encoded, map Framing.fields records) observed
        (prefix, execution@(consumed : _)) -> do
            _ <- Framing.readiness (prefix ++ [consumed])
            (group, remaining) <- Framing.takeGroup execution
            unless (null remaining) (Left "Unexpected records after the finite batch observation")
            admitMember planned bound (encoded, group)

admitGroup :: Infer.Plan -> V.Binding -> Framing.Group -> Either String Report
admitGroup planned bound group = admitMember planned bound (Framing.source group, group)

admitMember :: Infer.Plan -> V.Binding -> (ByteString, Framing.Group) -> Either String Report
admitMember planned bound (source, group) = do
    let matching = filter ((== Just (Wire.bindingValue bound)) . Fields.lookup "binding" . Framing.fields . Framing.consumed) (Framing.members group)
    case matching of
        [member] -> do
            let loaded = Framing.fields (Framing.loaded member)
                consumed = Framing.fields (Framing.consumed member)
                output = Framing.result member
            check planned bound (source, map Framing.fields [Framing.loaded member, Framing.consumed member, output]) (loaded, consumed, Framing.fields output, Framing.raw output)
        _ -> Left "Expected one declared member in the finite batch observation"

check :: Infer.Plan -> V.Binding -> (ByteString, [Object]) -> (Object, Object, Object, ByteString) -> Either String Report
check planned bound (source, values) (loaded, consumed, output, rawOutput) = do
    call <- either (Left . show) Right (Invocation.prepare bound planned)
    expected <- Json.decode (Invocation.batchInput call) >>= parseEither (withObject "expected inference consumption" pure)
    unless (Fields.delete "stage" consumed == expected) (Left "Inference consumption differs from the declared invocation")
    parseEither (checkLoad (planned, bound, expected)) loaded
    parseEither (checkResult bound) output
    observed <- either (Left . show) Right (Result.observeObjects planned values)
    Output.rawBehavior rawOutput (Result.behaviorBits observed)
    let digest = Artifact.hex (SHA256.hash source)
    length digest `seq` pure (Report digest bound observed loaded)

trace :: [(ByteString, Object)] -> Either String (Object, Object, Object, ByteString)
trace records = do
    let (prefix, execution) = span (\(_, fields) -> Fields.lookup "stage" fields `elem` map (Just . String) ["loading", "profile", "load"]) records
    mapM_ (parseEither (\fields -> when (Fields.member "phase" fields) (fail "Unexpected inference phase")) . snd) prefix
    case execution of
        [(_, loaded), (_, consumed), (_, measured), (raw, output)] -> do
            mapM_ (uncurry stage) [("loaded_adapter", loaded), ("consumed", consumed), ("inference", measured), ("result", output)]
            pure (loaded, consumed, output, raw)
        _ -> Left "Expected one ordered load, consumption, inference and result observation"
  where
    stage expected fields = do
        actual <- parseEither (.: "stage") fields
        unless (actual == (expected :: Text) && not (Fields.member "phase" fields)) (Left "Missing or reordered inference observation stage")

checkLoad :: (Infer.Plan, V.Binding, Object) -> Object -> Parser ()
checkLoad (planned, bound, expected) fields = do
    let required = ["stage", "binding", "load", "image", "requested", "consumed", "tokenizer", "base", "assembly", "model", "revision"]
        selected = Infer.requested planned
        image = Infer.image selected
        imageValue = object ["artifact" .= Bytes.unpack (Load.artifact image), "profile" .= Bytes.unpack (Load.profile image)]
    Json.fields (required ++ ["scope" | Fields.member "scope" fields]) fields
    checkBinding bound fields
    loading <- fields .: "load"
    unless (Just loading == Fields.lookup "load" expected) (fail "Inference load invocation differs from consumption")
    actual <- fields .: "image"
    unless (actual == imageValue) (fail "Inference load image differs from the declared materialization")
    mapM_ (identity fields) [("requested", Infer.artifact selected), ("consumed", Infer.artifact selected), ("tokenizer", Infer.tokenizer selected), ("base", Infer.base selected), ("assembly", Infer.assembly selected)]
    mapM_ (nonempty fields) (["model", "revision"] ++ ["scope" | Fields.member "scope" fields])

identity :: Object -> (Key, String) -> Parser ()
identity fields (key, expected) = do
    actual <- fields .: key >>= Json.identity
    unless (actual == expected) (fail "Inference load materialization differs from the declared request")

nonempty :: Object -> Key -> Parser ()
nonempty fields key = do
    value <- fields .: key
    when (Text.null value) (fail "Expected nonempty inference model or scope text")

checkResult :: V.Binding -> Object -> Parser ()
checkResult bound fields = do
    Json.fields ["stage", "binding", "adapter", "tokenizer", "base", "assembly", "request", "tokens", "prompt_length", "behavior", "behavior_bits", "text", "truncated"] fields
    checkBinding bound fields
    fields .: "request" >>= withObject "inference numerical request" (Json.fields ["prompt", "tokens", "temperature", "seed"])

checkBinding :: V.Binding -> Object -> Parser ()
checkBinding bound fields = do
    actual <- fields .: "binding"
    unless (actual == Wire.bindingValue bound) (fail "Inference observation binding mismatch")

result :: Report -> Result.Result
result (Report _ _ observed _) = observed

binding :: Report -> V.Binding
binding (Report _ bound _ _) = bound

logDigest :: Report -> String
logDigest (Report digest _ _ _) = digest

policyDescription :: Report -> Either String Policy.Description
policyDescription (Report _ _ observed loaded) = do
    source <- parseEither (\fields -> (,) <$> fields .: "model" <*> fields .: "revision") loaded
    let requested = Result.consumed observed
    Policy.describe source (Infer.artifact requested, Infer.tokenizer requested, Infer.base requested, Infer.assembly requested)

describe :: Report -> Value
describe report@(Report _ _ observed loaded) =
    object
        [ "log_sha256" .= logDigest report
        , "binding" .= Wire.bindingValue (binding report)
        , "tokens" .= Result.tokens observed
        , "behavior_bits" .= Result.behaviorBits observed
        , "prompt_length" .= Result.promptLength observed
        , "text" .= Result.response observed
        , "truncated" .= Result.truncated observed
        , "model" .= Fields.lookup "model" loaded
        , "revision" .= Fields.lookup "revision" loaded
        , "adapter" .= Infer.artifact requested
        , "tokenizer" .= Infer.tokenizer requested
        , "base" .= Infer.base requested
        , "assembly" .= Infer.assembly requested
        , "request" .= object ["prompt" .= Infer.prompt requested, "tokens" .= Infer.tokens requested, "temperature" .= Infer.temperature requested, "seed" .= Infer.seed requested]
        ]
  where
    requested = Result.consumed observed
