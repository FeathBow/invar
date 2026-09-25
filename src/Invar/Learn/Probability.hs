{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Probability (Sample (..), validate, observe) where

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
import GHC.Float (castDoubleToWord64)
import Invar.Float32 qualified as Float32
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

data Sample = Sample {sampleObject :: Object, sampleName :: Text, behavior :: [Word32], proximal :: [Word32], fixed :: [Word32], current :: [Word32], linearized :: [Word32]}

observe :: (Value, Value, ByteString) -> ByteString -> Either String [Sample]
observe (intended, consumed, completed) encoded = do
    output <- Json.decode completed
    reported <- Json.floatingAt ["update", "loss"] completed
    Json.decode encoded >>= parseEither (withObject "probability observation" (document (intended, consumed, output) reported))

document :: (Value, Value, Value) -> Double -> Object -> Parser [Sample]
document (intended, consumed, output) reported fields = do
    exact ["format", "invocation", "request", "samples", "scalar_reference", "loss"] fields
    format <- fields .: "format"
    engine <- case format :: Text of
        "invar-probabilities-v2" -> pure False
        "invar-probabilities-v3" -> pure True
        _ -> fail "Unknown probability observation format"
    reference <- fields .: "scalar_reference"
    unless (reference == Objective.reference) (fail "Unknown scalar objective reference")
    invocation <- fields .: "invocation"
    unless (invocation == intended) (fail "Probability observation invocation mismatch")
    request <- fields .: "request"
    unless (request == consumed) (fail "Probability observation request mismatch")
    _ <- Request.parse request
    observations <- fields .: "samples"
    (expectedLoss, checked) <- withObject "probability request" (samples engine observations) request
    observedLoss <- fields .: "loss"
    unless (observedLoss == expectedLoss) (fail "Reported loss differs from the core token mean")
    summary <- withObject "update result" (.: "update") output
    unless (castDoubleToWord64 reported == Float32.widened expectedLoss) (fail "Update summary differs from the core scalar loss")
    tokens <- summary .: "active_tokens" :: Parser Integer
    unless (tokens == fromIntegral (sum (map (length . behavior) checked))) (fail "Update summary differs from the scalar observation token count")
    pure checked

samples :: Bool -> [Object] -> Object -> Parser (Word32, [Sample])
samples engine observed request = do
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
    checked <- traverse (check (profile, sum counts) (Map.intersectionWith (,) expected advantages)) (zip names observed)
    loss <- either (fail . show) pure (Objective.mean32 (concatMap fst checked))
    pure (loss, map snd checked)
  where
    reward item = Advantage.Reward <$> item .: "sample" <*> item .: "group" <*> item .: "reward"
    check settings expected (name, item) = case Map.lookup name expected of
        Just (original, advantage) -> sample engine settings advantage (original, item)
        _ -> fail "Unknown probability sample"

sample :: Bool -> (Objective.Profile, Int) -> Word32 -> (Object, Object) -> Parser ([Word32], Sample)
sample engine settings advantage (original, fields) = do
    exact (["sample", "dtype", "active", "objective"] ++ roles ++ ["linearized" | engine]) fields
    name <- fields .: "sample"
    dtype <- fields .: "dtype"
    unless (dtype == ("F32" :: Text)) (fail "Update probability observations must use FP32")
    consumed <- original .: "behavior_bits" :: Parser [Word32]
    [behaviorWords, proximalWords, fixedWords, currentWords, advantageWords] <- traverse (vector (length consumed) fields) roles
    unless (behaviorWords == consumed) (fail "Behavior probability words differ from consumed input")
    linearizedWords <- if engine then vector (length consumed) fields "linearized" else pure proximalWords
    unless (not engine || (proximalWords == behaviorWords && currentWords == behaviorWords)) (fail "Proximal and current probability words differ from the engine's behavior words")
    scored <- if engine then original .: "reference_bits" else pure []
    unless (not engine || fixedWords == (if null scored then behaviorWords else scored)) (fail "Reference probability words differ from the engine's reference scores")
    claimed <- original .: "advantage_bits"
    unless (claimed == advantage) (fail "Consumed advantage expectation differs from the core reference")
    unless (advantageWords == replicate (length consumed) advantage) (fail "Actual advantage words differ from the core reference")
    active <- fields .: "active" :: Parser [Bool]
    unless (active == replicate (length consumed) True) (fail "Update probability active mask mismatch")
    let inputs = [Objective.Inputs {Objective.behavior = b, Objective.proximal = p, Objective.fixed = q, Objective.current = c, Objective.advantage = a} | (b, p, q, c, a) <- zip5 behaviorWords proximalWords fixedWords currentWords advantageWords]
    terms <- objective settings fields inputs
    pure (terms, Sample fields name behaviorWords proximalWords fixedWords currentWords linearizedWords)

objective :: (Objective.Profile, Int) -> Object -> [Objective.Inputs] -> Parser [Word32]
objective (profile, count) fields inputs = do
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

vector :: Int -> Object -> Key -> Parser [Word32]
vector count fields role = do
    encoded <- fields .: role :: Parser [Word32]
    unless (length encoded == count && count > 0) (fail "Probability vector token count mismatch")
    let valid = if role == "advantage" then Float32.finite else Float32.logProbability
    unless (all valid encoded) (fail "Invalid probability or advantage floating-point words")
    pure encoded

exact :: [Key] -> Object -> Parser ()
exact names fields = unless (Set.fromList (Fields.keys fields) == Set.fromList names) (fail "Unexpected probability observation fields")
