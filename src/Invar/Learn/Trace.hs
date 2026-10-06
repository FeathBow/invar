{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Trace (Attempt (..), invoked, reports, attempt, readiness, completion) where

import Control.Monad (unless, when)
import Data.Aeson (Object, Value (..), object, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text qualified as Text
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Learn qualified as Learn
import Invar.Learn.Protocol qualified as Protocol
import Invar.Learn.Report qualified as Report
import Invar.Learn.Request qualified as Request
import Invar.Learn.Step qualified as Step
import Invar.Learn.Stream qualified as S
import Invar.Load qualified as Load
import Invar.Materialization qualified as Materialization
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as Image

data Attempt = Attempt {opening :: String, stream :: S.Stream, result :: Maybe Object, stopped :: Maybe String}

invoked :: Report.Report -> Either String (V.Binding, Text.Text, Value)
invoked report = do
    bound <- parseEither (withObject "update invocation" Wire.binding) (Report.invocation report)
    program <- parseEither (withObject "update invocation" (.: "program")) (Report.invocation report)
    pure (bound, program, Report.request report)

reports :: Report.Report -> Attempt -> Either String ()
reports report attempted = case (result attempted, stopped attempted) of
    (Just finished, _) -> unless (Object finished == Report.result report) (Left "Update trace result differs from the admitted report")
    (Nothing, Just problem) -> Left problem
    (Nothing, Nothing) -> Left "Incomplete learner execution trace"

attempt :: Learn.Settings -> (V.Binding, Text.Text, Value) -> [Object] -> Either String Attempt
attempt settings (bound, program, request) events = do
    let selected = Materialization.learning (Learn.policy settings, Learn.learner settings, Learn.tokenizer settings, Learn.base settings, Learn.assembly settings, Learn.reference settings)
    pair@(_, consumed) <- readiness request (take 2 events)
    unless (Fields.lookup "program" consumed == Just (String program) && Fields.lookup "request" consumed == Just request) (Left "Learner consumption differs from the declared update")
    declared <- parseEither (withObject "update request" pure) request
    bindings selected (bound, declared) pair
    checked <- parseEither Request.parse request
    let (reported, rest) = span stepping (drop 2 events)
    (begun, _) <- first show (Protocol.partial bound checked [])
    (advanced, broken) <- first show (Protocol.partial bound checked reported)
    let attempted = Attempt (S.opening begun) advanced
    pure $ case (broken, rest) of
        (Just problem, _) -> attempted Nothing (Just (show problem))
        (Nothing, []) -> attempted Nothing Nothing
        (Nothing, updated : remaining) -> either (attempted Nothing . Just) (`attempted` Nothing) (concluding checked advanced updated remaining)
  where
    concluding checked advanced updated remaining = do
        stage "reward_update" updated
        let (staged, ending) = span staging remaining
        mapM_ (\fields -> when (Fields.member "phase" fields) (Left "Unexpected learner observation stage")) staged
        case ending of
            [] -> Right Nothing
            [finished] -> do
                completion finished
                first show (Protocol.validateResult bound checked advanced finished)
                Right (Just finished)
            _ -> Left "Output follows the learner result"
    staging fields = Fields.lookup "stage" fields `elem` map (Just . String) ["checkpoint", "artifacts"]
    stepping fields = Fields.lookup "stage" fields `elem` map (Just . String) Step.stages

readiness :: Value -> [Object] -> Either String (Object, Object)
readiness _ events = case events of
    [loaded, consumed] -> do
        stage "loaded_learner" loaded
        parseEither (Json.fields ["stage", "binding", "state", "load", "image", "model", "revision"]) loaded
        mapM_ (\key -> parseEither (\fields -> fields .: key >>= \value -> when (Text.null value) (fail "Expected learner model and revision")) loaded) ["model", "revision"]
        stage "consumed" consumed
        parseEither (Json.fields ["stage", "binding", "program", "request", "load"]) consumed
        pure (loaded, consumed)
    _ -> Left "Expected one learner load followed by one update consumption"

completion :: Object -> Either String ()
completion finished = do
    stage "result" finished
    parseEither (Json.fields ["stage", "binding", "request", "update", "gradients", "probabilities", "adapter", "learner", "storage"]) finished
    parseEither (\fields -> fields .: "update" >>= withObject "update summary" (Json.fields ["gradient_norm", "reward_gradient_norm", "active_tokens", "before", "after", "nonzero_advantages"])) finished

stage :: Text.Text -> Object -> Either String ()
stage expected fields = unless (Fields.lookup "stage" fields == Just (String expected) && not (Fields.member "phase" fields)) (Left "Unexpected learner observation stage")

bindings :: Image.Image -> (V.Binding, Object) -> (Object, Object) -> Either String ()
bindings selected (bound, input) (loaded, consumed) = do
    let image = object ["artifact" .= Bytes.unpack (Image.artifact selected), "profile" .= Bytes.unpack (Image.profile selected)]
        state = Object (Fields.filterWithKey (\key _ -> key `elem` ["policy", "learner", "reference", "tokenizer", "base", "assembly", "optimizer"]) input)
    planned <- first show (Load.prepare bound selected)
    let loading = Wire.invocationValue bound (Load.program planned)
    mapM_ (\fields -> unless (Fields.lookup "load" fields == Just loading && Fields.lookup "binding" fields == Just (Wire.bindingValue bound)) (Left "Learner load invocation differs from the declared update")) [loaded, consumed]
    unless (Fields.lookup "state" loaded == Just state && Fields.lookup "image" loaded == Just image) (Left "Learner loaded state or image differs from the declared update")
