{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Probability (validate, observe) where

import Control.Monad (unless)
import Data.Aeson (Object, Value, withObject, (.:))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.List (zip5)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Word (Word32)
import GHC.Float (castDoubleToWord64, castWord32ToFloat, float2Double)
import Invar.Infer.Wire qualified as Binding
import Invar.Json qualified as Json
import Invar.Learn.Advantage qualified as Advantage
import Invar.Learn.Objective qualified as Objective
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Request qualified as Request
import Invar.Spec.Invocation qualified as V

validate :: P.Result -> ByteString -> Either String ()
validate expected encoded = do
    let completed = P.completion expected
        intended = Binding.invocationValue (V.completedBinding completed) (V.completedProgram completed)
    _ <- observe (intended, P.request expected, V.completedOutput completed) encoded
    pure ()

observe :: (Value, Value, ByteString) -> ByteString -> Either String [Object]
observe (intended, consumed, completed) encoded = do
    output <- Json.decode completed
    reported <- Json.floatingAt ["update", "loss"] completed
    Json.decode encoded >>= parseEither (withObject "probability observation" (document (intended, consumed, output) reported))

document :: (Value, Value, Value) -> Double -> Object -> Parser [Object]
document (intended, consumed, output) reported fields = do
    exact ["format", "invocation", "request", "samples", "scalar_reference", "loss"] fields
    format <- fields .: "format"
    unless (format == ("invar-probabilities-v2" :: Text)) (fail "Unknown probability observation format")
    reference <- fields .: "scalar_reference"
    unless (reference == Objective.reference) (fail "Unknown scalar objective reference")
    invocation <- fields .: "invocation"
    unless (invocation == intended) (fail "Probability observation invocation mismatch")
    request <- fields .: "request"
    unless (request == consumed) (fail "Probability observation request mismatch")
    _ <- Request.parse request
    observations <- fields .: "samples"
    expectedLoss <- withObject "probability request" (samples observations) request
    observedLoss <- fields .: "loss"
    unless (observedLoss == expectedLoss) (fail "Reported loss differs from the core token mean")
    summary <- withObject "update result" (.: "update") output
    unless (castDoubleToWord64 reported == castDoubleToWord64 (float2Double (castWord32ToFloat expectedLoss))) (fail "Update summary differs from the core scalar loss")
    counts <- traverse (\item -> length <$> (item .: "active" :: Parser [Bool])) observations
    tokens <- summary .: "active_tokens" :: Parser Integer
    unless (tokens == fromIntegral (sum counts)) (fail "Update summary differs from the scalar observation token count")
    pure observations

samples :: [Object] -> Object -> Parser Word32
samples observed request = do
    order <- request .: "order" :: Parser [String]
    delivered <- request .: "samples" :: Parser [Object]
    named <- traverse (\item -> (,) <$> item .: "sample" <*> pure item) delivered
    delta <- request .: "delta"
    rewards <- traverse reward delivered
    advantages <- either (fail . show) pure (Advantage.calculate delta rewards)
    names <- traverse (.: "sample") observed
    unless (names == order) (fail "Probability samples differ from logical order")
    let expected = Map.fromList named
    profile <- Objective.Profile <$> request .: "epsilon" <*> request .: "penalty"
    counts <- traverse (\item -> length <$> (item .: "behavior_bits" :: Parser [Word32])) delivered
    terms <- traverse (check (profile, sum counts) (Map.intersectionWith (,) expected advantages)) (zip names observed)
    either (fail . show) pure (Objective.mean32 (concat terms))
  where
    reward item = Advantage.Reward <$> item .: "sample" <*> item .: "group" <*> item .: "reward"
    check settings expected (name, item) = case Map.lookup name expected of
        Just (original, advantage) -> sample settings advantage (original, item)
        _ -> fail "Unknown probability sample"

sample :: (Objective.Profile, Int) -> Word32 -> (Object, Object) -> Parser [Word32]
sample settings advantage (original, fields) = do
    exact (["sample", "dtype", "active", "objective"] ++ roles) fields
    dtype <- fields .: "dtype"
    unless (dtype == ("F32" :: Text)) (fail "Update probability observations must use FP32")
    behavior <- original .: "behavior_bits" :: Parser [Word32]
    actual <- fields .: "behavior"
    unless (actual == behavior) (fail "Behavior probability words differ from consumed input")
    mapM_ (vector (length behavior) fields) roles
    claimed <- original .: "advantage_bits"
    unless (claimed == advantage) (fail "Consumed advantage expectation differs from the core reference")
    values <- fields .: "advantage"
    unless (values == replicate (length behavior) advantage) (fail "Actual advantage words differ from the core reference")
    active <- fields .: "active" :: Parser [Bool]
    unless (active == replicate (length behavior) True) (fail "Update probability active mask mismatch")
    objective settings fields

objective :: (Objective.Profile, Int) -> Object -> Parser [Word32]
objective (profile, count) fields = do
    behavior <- fields .: "behavior"
    proximal <- fields .: "proximal"
    reference <- fields .: "reference"
    current <- fields .: "current"
    advantage <- fields .: "advantage"
    let inputs = [Objective.Inputs {Objective.behavior = b, Objective.proximal = p, Objective.fixed = q, Objective.current = c, Objective.advantage = a} | (b, p, q, c, a) <- zip5 behavior proximal reference current advantage]
    expected <- either (fail . show) pure (Objective.calculate profile count inputs)
    actual <- fields .: "objective"
    let outputs = [("terms", map Objective.term expected), ("current_gradient", map Objective.gradient expected), ("reward_gradient", map Objective.rewardGradient expected)]
    exact (map fst outputs) actual
    mapM_ (check actual) outputs
    pure (map Objective.term expected)
  where
    check observed (name, expected) = do
        actual <- observed .: name
        unless (actual == expected) (fail ("Actual scalar " ++ show name ++ " differs from the core reference"))

roles :: [Key]
roles = ["behavior", "proximal", "reference", "current", "advantage"]

vector :: Int -> Object -> Key -> Parser ()
vector count fields role = do
    encoded <- fields .: role :: Parser [Word32]
    unless (length encoded == count && count > 0) (fail "Probability vector token count mismatch")
    let valid word = let number = castWord32ToFloat word in not (isNaN number || isInfinite number) && (role == "advantage" || number <= 0)
    unless (all valid encoded) (fail "Invalid probability or advantage floating-point words")

exact :: [Key] -> Object -> Parser ()
exact names fields = unless (Set.fromList (Fields.keys fields) == Set.fromList names) (fail "Unexpected probability observation fields")
