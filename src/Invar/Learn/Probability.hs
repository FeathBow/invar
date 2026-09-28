{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Probability (Sample (..), validate, observe) where

import Control.Monad (unless)
import Data.Aeson (Object, Value, withObject, (.:))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.List (sortOn, zip4)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Word (Word32)
import GHC.Float (castDoubleToWord64)
import Invar.Float32 qualified as Float32
import Invar.Infer.Wire qualified as Binding
import Invar.Json qualified as Json
import Invar.Learn.Advantage qualified as Advantage
import Invar.Learn.Objective qualified as Objective
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Request qualified as Request
import Invar.Learn.Stream qualified as S
import Invar.Spec.Invocation qualified as V
import Numeric.Natural (Natural)

validate :: P.Result -> ByteString -> Either String ()
validate expected encoded = do
    let completed = P.completion expected
        intended = Binding.invocationValue (V.completedBinding completed) (V.completedProgram completed)
        streamed = P.stream expected
    observed <- observe (intended, P.checkedRequest expected, V.completedOutput completed) encoded
    unless (Map.fromList [(sampleName entry, proximal entry) | entry <- observed] == S.proximals streamed) (Left "Probability artifact proximal words differ from the reported steps")
    unless (sortOn (\(position, name, _) -> (position, name)) [(position, sampleName entry, words32) | entry <- observed, (position, words32) <- currents entry] == sortOn (\(position, name, _) -> (position, name)) (S.currents streamed)) (Left "Probability artifact current words differ from the reported steps")

data Sample = Sample {sampleObject :: Object, sampleName :: Text, behavior :: [Word32], proximal :: [Word32], fixed :: [Word32], currents :: [(Natural, [Word32])]}

observe :: (Value, Request.Request, ByteString) -> ByteString -> Either String [Sample]
observe (intended, consumed, completed) encoded = do
    output <- Json.decode completed
    reported <- Json.floatingAt ["update", "loss"] completed
    Json.decode encoded >>= parseEither (withObject "probability observation" (document (intended, consumed, output) reported))

document :: (Value, Request.Request, Value) -> Double -> Object -> Parser [Sample]
document (intended, consumed, output) reported fields = do
    exact ["format", "invocation", "request", "samples", "scalar_reference", "losses"] fields
    format <- fields .: "format"
    unless (format == ("invar-probabilities-v4" :: Text)) (fail "Unknown probability observation format")
    reference <- fields .: "scalar_reference"
    unless (reference == Objective.reference) (fail "Unknown scalar objective reference")
    invocation <- fields .: "invocation"
    unless (invocation == intended) (fail "Probability observation invocation mismatch")
    request <- fields .: "request"
    unless (request == Request.value consumed) (fail "Probability observation request mismatch")
    observations <- fields .: "samples"
    (expectedLosses, checked) <- withObject "probability request" (samples observations) (Request.value consumed)
    observedLosses <- fields .: "losses"
    unless (observedLosses == expectedLosses) (fail "Reported step losses differ from the core token means")
    summary <- withObject "update result" (.: "update") output
    case expectedLosses of
        firstLoss : _ -> unless (castDoubleToWord64 reported == Float32.widened firstLoss) (fail "Update summary differs from the core scalar loss of the first step")
        [] -> fail "An update needs at least one step loss"
    tokens <- summary .: "active_tokens" :: Parser Integer
    unless (tokens == fromIntegral (sum (map (length . behavior) checked))) (fail "Update summary differs from the scalar observation token count")
    pure checked

samples :: [Object] -> Object -> Parser ([Word32], [Sample])
samples observed request = do
    order <- request .: "order" :: Parser [Text]
    plan <- request .: "steps" :: Parser [[Text]]
    delivered <- request .: "samples" :: Parser [Object]
    named <- traverse (\item -> (,) <$> item .: "sample" <*> pure item) delivered
    delta <- request .: "delta"
    rewards <- traverse reward delivered
    advantages <- either (fail . show) pure (Advantage.calculate delta rewards)
    names <- traverse (.: "sample") observed
    unless (names == order) (fail "Probability samples differ from logical order")
    profile <- Objective.Profile <$> request .: "epsilon" <*> request .: "penalty"
    let originals = Map.fromList named
        lengths = Map.map (\item -> either (const 0) length (parseEither (.: "behavior_bits") item :: Either String [Word32])) originals
        denominators = [fromIntegral (sum [Map.findWithDefault 0 name lengths | name <- batch]) | batch <- plan]
        participation name = [position | (position, batch) <- zip [0 ..] plan, name `elem` batch]
    checked <- traverse (\(name, item) -> maybe (fail "Unknown probability sample") (\(original, advantage) -> sample (profile, denominators, participation name, firstStep plan) advantage (original, item)) (Map.lookup name (Map.intersectionWith (,) originals (Map.mapKeys Text.pack advantages)))) (zip names observed)
    let terms = Map.fromList [((position, sampleName entry), words32) | (termsBySample, entry) <- checked, (position, words32) <- termsBySample]
    ordered <- traverse (\(position, batch) -> concat <$> traverse (\name -> maybe (fail "Probability artifact misses a declared step") pure (Map.lookup (position, name) terms)) batch) (zip [0 :: Natural ..] plan)
    losses <- either (fail . show) pure (traverse Objective.mean32 ordered)
    pure (losses, map snd checked)
  where
    reward item = Advantage.Reward <$> item .: "sample" <*> item .: "group" <*> item .: "reward"
    firstStep plan = case plan of
        batch : _ -> batch
        [] -> []

sample :: (Objective.Profile, [Natural], [Natural], [Text]) -> Word32 -> (Object, Object) -> Parser ([(Natural, [Word32])], Sample)
sample (profile, denominators, participation, first) advantage (original, fields) = do
    exact ["sample", "dtype", "behavior", "reference", "advantage", "proximal", "steps"] fields
    name <- fields .: "sample"
    dtype <- fields .: "dtype"
    unless (dtype == ("F32" :: Text)) (fail "Update probability observations must use FP32")
    consumed <- original .: "behavior_bits" :: Parser [Word32]
    behaviorWords <- vector (length consumed) fields "behavior"
    unless (behaviorWords == consumed) (fail "Behavior probability words differ from consumed input")
    fixedWords <- vector (length consumed) fields "reference"
    scored <- original .: "reference_bits"
    unless (fixedWords == (if null scored then behaviorWords else scored)) (fail "Reference probability words differ from the engine's reference scores")
    claimed <- original .: "advantage_bits"
    unless (claimed == advantage) (fail "Consumed advantage expectation differs from the core reference")
    reportedAdvantage <- fields .: "advantage"
    unless (reportedAdvantage == advantage) (fail "Actual advantage word differs from the core reference")
    proximalWords <- vector (length consumed) fields "proximal"
    entries <- fields .: "steps" :: Parser [Object]
    positions <- traverse (.: "step") entries
    unless (positions == participation) (fail "Probability steps differ from the declared mini-batches of this sample")
    observed <- traverse (entry (length consumed) (behaviorWords, proximalWords, fixedWords)) (zip entries positions)
    case observed of
        (0, (currentWords, _)) : _ | name `elem` first -> unless (currentWords == proximalWords) (fail "Proximal words differ from the first step's observation")
        _ -> pure ()
    pure ([(position, termsWords) | (position, (_, termsWords)) <- observed], Sample fields name behaviorWords proximalWords fixedWords [(position, currentWords) | (position, (currentWords, _)) <- observed])
  where
    entry count roles (item, position) = do
        exact ["step", "current", "objective"] item
        currentWords <- vector count item "current"
        let (behaviorWords, proximalWords, fixedWords) = roles
            inputs = [Objective.Inputs {Objective.behavior = b, Objective.proximal = p, Objective.fixed = q, Objective.current = c, Objective.advantage = advantage} | (b, p, q, c) <- zip4 behaviorWords proximalWords fixedWords currentWords]
        denominator <- maybe (fail "Unknown probability step") pure (lookupStep position denominators)
        termsWords <- objective (profile, fromIntegral denominator) item inputs
        pure (position, (currentWords, termsWords))
    lookupStep position values = case drop (fromIntegral position) values of
        value : _ -> Just value
        [] -> Nothing

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

vector :: Int -> Object -> Key -> Parser [Word32]
vector count fields role = do
    encoded <- fields .: role :: Parser [Word32]
    unless (length encoded == count && count > 0) (fail "Probability vector token count mismatch")
    unless (all Float32.logProbability encoded) (fail "Invalid probability or advantage floating-point words")
    pure encoded

exact :: [Key] -> Object -> Parser ()
exact names fields = unless (Set.fromList (Fields.keys fields) == Set.fromList names) (fail "Unexpected probability observation fields")
