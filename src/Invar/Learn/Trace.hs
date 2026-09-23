{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Trace (validate, validateObserved, readiness, completion) where

import Control.Monad (unless, when)
import Data.Aeson (Object, Value (..), object, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.Bifunctor (first)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text qualified as Text
import Data.Word (Word32)
import GHC.Float (castFloatToWord32)
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Learn qualified as Learn
import Invar.Learn.Protocol qualified as Protocol
import Invar.Learn.Report qualified as Report
import Invar.Load qualified as Load
import Invar.Materialization qualified as Materialization
import Invar.Spec.Load qualified as Image

-- The caller supplies the complete load-to-result segment, without diagnostics
-- preceding the load.
validate :: Learn.Settings -> Report.Report -> [Object] -> Either String ()
validate settings = validateWith (Materialization.learning (Learn.policy settings, Learn.learner settings, Learn.tokenizer settings, Learn.base settings, Learn.assembly settings, Learn.reference settings))

validateObserved :: Report.Report -> [Object] -> Either String ()
validateObserved report events = do
    selected <- parseEither (withObject "observed update materialization" materialization) (Report.request report)
    validateWith selected report events
  where
    materialization input = Materialization.learning <$> ((,,,,,) <$> input .: "policy" <*> input .: "learner" <*> input .: "tokenizer" <*> input .: "base" <*> input .: "assembly" <*> input .: "reference")

validateWith :: Image.Image -> Report.Report -> [Object] -> Either String ()
validateWith selected report events = case reverse events of
    result : remaining | (staged, updated : preceding) <- span staging remaining -> do
        pair <- readiness (Report.request report) (reverse preceding)
        bindings selected report pair
        stage "reward_update" updated
        mapM_ (\fields -> when (Fields.member "phase" fields) (Left "Unexpected learner observation stage")) staged
        completion (Report.request report) result
        unless (Object result == Report.result report) (Left "Update trace result differs from the admitted report")
    _ -> Left "Incomplete learner execution trace"
  where
    staging fields = Fields.lookup "stage" fields `elem` map (Just . String) ["checkpoint", "artifacts"]

readiness :: Value -> [Object] -> Either String (Object, Object)
readiness request events = case events of
    loaded : measured : remaining -> do
        stage "loaded_learner" loaded
        stage "probability_roles" measured
        parseEither (Json.fields ["stage", "binding", "state", "load", "image", "model", "revision"]) loaded
        mapM_ (\key -> parseEither (\fields -> fields .: key >>= \value -> when (Text.null value) (fail "Expected learner model and revision")) loaded) ["model", "revision"]
        let (roles, suffix) = span (\fields -> Fields.lookup "stage" fields == Just (String "roles")) remaining
        input <- parseEither (withObject "update request" pure) request
        samples <- parseEither (.: "samples") input
        unless (length roles == length (samples :: [Object])) (Left "Update role observation inventory mismatch")
        mapM_ (uncurry (role input)) (zip samples roles)
        case suffix of
            [consumed] -> do
                stage "consumed" consumed
                parseEither (Json.fields ["stage", "binding", "program", "request", "load"]) consumed
                pure (loaded, consumed)
            _ -> Left "Expected one ordered update consumption after all probability roles"
    _ -> Left "Incomplete learner readiness trace"

completion :: Value -> Object -> Either String ()
completion request result = do
    stage "result" result
    parseEither (Json.fields ["stage", "binding", "request", "update", "gradients", "probabilities", "adapter", "learner", "storage"]) result
    parseEither (\fields -> fields .: "update" >>= withObject "update summary" (Json.fields ["loss", "gradient_norm", "reward_gradient_norm", "active_tokens", "before", "after", "nonzero_advantages"])) result
    first show (Protocol.validateSummary request result)

stage :: Text.Text -> Object -> Either String ()
stage expected fields = unless (Fields.lookup "stage" fields == Just (String expected) && not (Fields.member "phase" fields)) (Left "Unexpected learner observation stage")

bindings :: Image.Image -> Report.Report -> (Object, Object) -> Either String ()
bindings selected report (loaded, consumed) = do
    bound <- parseEither (withObject "update invocation" Wire.binding) (Report.invocation report)
    input <- parseEither (withObject "update request" pure) (Report.request report)
    let image = object ["artifact" .= Bytes.unpack (Image.artifact selected), "profile" .= Bytes.unpack (Image.profile selected)]
        state = Object (Fields.filterWithKey (\key _ -> key `elem` ["policy", "learner", "reference", "tokenizer", "base", "assembly", "optimizer"]) input)
    planned <- first show (Load.prepare bound selected)
    let loading = Wire.invocationValue bound (Load.program planned)
    mapM_ (\fields -> unless (Fields.lookup "load" fields == Just loading && Fields.lookup "binding" fields == Just (Wire.bindingValue bound)) (Left "Learner load invocation differs from the declared update")) [loaded, consumed]
    unless (Fields.lookup "state" loaded == Just state && Fields.lookup "image" loaded == Just image) (Left "Learner loaded state or image differs from the declared update")

role :: Object -> Object -> Object -> Either String ()
role request sample fields = do
    parseEither (Json.fields ["stage", "sample", "proximal_policy", "reference_policy", "proximal", "reference", "advantage"]) fields
    unless (Fields.lookup "sample" fields == Fields.lookup "sample" sample && Fields.lookup "proximal_policy" fields == Fields.lookup "policy" request && Fields.lookup "reference_policy" fields == Fields.lookup "reference" request) (Left "Probability role observation binding mismatch")
    values <- traverse (\key -> parseEither (\value -> (value .: key :: Parser [Value]) >>= traverse Json.finite) fields) ["proximal", "reference"]
    count <- parseEither (\value -> length <$> (value .: "behavior_bits" :: Parser [Value])) sample
    unless (all ((== count) . length) values) (Left "Probability role observation token inventory mismatch")
    advantage <- parseEither (\value -> value .: "advantage" >>= Json.finite) fields
    expected <- parseEither (.: "advantage_bits") sample :: Either String Word32
    unless (castFloatToWord32 (realToFrac advantage) == expected) (Left "Probability role advantage differs from the consumed word")
