{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Observation (Report, view, admitFrames, admitGroup, result, binding, logDigest, policyDescription, describe) where

import Control.Monad (unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Object, Value (..), object, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.Text (Text)
import Invar.Artifact qualified as Artifact
import Invar.Infer qualified as Infer
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Invocation qualified as Invocation
import Invar.Infer.Output qualified as Output
import Invar.Infer.Records qualified as Records
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Result qualified as Result
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Policy qualified as Policy
import Invar.Spec.Invocation qualified as V

data Report = Report String V.Binding Result.Result (String, String)
    deriving (Eq, Show)

view :: Replay.Logged -> Report
view logged = Report (Replay.source logged) (Trajectory.binding admitted) (Trajectory.result admitted) (Trajectory.model admitted, Trajectory.revision admitted)
  where
    admitted = Replay.trajectory logged

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
    Records.consumed expected consumed
    found <- Records.loaded (planned, bound, expected) loaded
    Records.result bound output
    observed <- either (Left . show) Right (Result.observeObjects planned values)
    Output.rawBehavior rawOutput (Result.behaviorBits observed)
    let digest = Artifact.hex (SHA256.hash source)
    length digest `seq` pure (Report digest bound observed (Records.model found, Records.revision found))

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

result :: Report -> Result.Result
result (Report _ _ observed _) = observed

binding :: Report -> V.Binding
binding (Report _ bound _ _) = bound

logDigest :: Report -> String
logDigest (Report digest _ _ _) = digest

policyDescription :: Report -> Either String Policy.Description
policyDescription (Report _ _ observed source) = do
    let requested = Result.consumed observed
    Policy.describe source (Infer.artifact requested, Infer.tokenizer requested, Infer.base requested, Infer.assembly requested)

describe :: Report -> Value
describe report@(Report _ _ observed (model, revision)) =
    object
        [ "log_sha256" .= logDigest report
        , "binding" .= Wire.bindingValue (binding report)
        , "tokens" .= Result.tokens observed
        , "behavior_bits" .= Result.behaviorBits observed
        , "prompt_length" .= Result.promptLength observed
        , "text" .= Result.response observed
        , "truncated" .= Result.truncated observed
        , "model" .= model
        , "revision" .= revision
        , "adapter" .= Infer.artifact requested
        , "tokenizer" .= Infer.tokenizer requested
        , "base" .= Infer.base requested
        , "assembly" .= Infer.assembly requested
        , "request" .= object ["prompt" .= Infer.prompt requested, "tokens" .= Infer.tokens requested, "temperature" .= Infer.temperature requested, "seed" .= Infer.seed requested]
        ]
  where
    requested = Result.consumed observed
