{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Wire (Error (..), lower, image) where

import Control.Monad (unless)
import Data.Aeson (Value, eitherDecode, encode, object, toJSON, (.=))
import Data.Aeson.Key qualified as Key
import Data.Aeson.Types (Pair, parseEither)
import Data.Bifunctor (first)
import Data.Char (chr, ord)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Ratio (denominator, numerator)
import Data.Set qualified as Set
import Invar.Learn.Advantage qualified as Advantage
import Invar.Learn.Algorithm qualified as Algorithm
import Invar.Learn.Request qualified as Request
import Invar.Materialization qualified as Materialization
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Load qualified as Load
import Invar.Spec.Value qualified as V
import Numeric.Natural (Natural)

data Error = Shape String | Membership String | Numerical Advantage.Error
    deriving (Eq, Show)

data Inputs = Inputs {trajectories :: Map Natural (V.Value Natural), generations :: Map Natural (V.Value Natural), behavior :: Map Natural (V.Value Natural), scores :: Map Natural (V.Value Natural), order :: [Natural]}

lower :: E.Emission -> Either Error Value
lower command = do
    (chosen, payload) <- emitted command
    _ <- image command
    samples <- inputs payload
    learner <- field "learner" payload
    algorithm <- field "algorithm" payload
    policy <- field "policy" learner >>= text
    checkpoint <- field "learner" learner >>= text
    tokenizer <- field "tokenizer" learner >>= text
    base <- field "base" learner >>= text
    assembly <- field "assembly" learner >>= text
    behavior <- field "behavior_model" payload >>= modelValue
    schedule <- field "schedule" payload >>= scheduleValue
    reference <- field "reference" payload >>= text
    source <- field "reference_source" payload >>= text
    optimizer <- field "optimizer" learner >>= optimizerValue
    epsilon <- field "epsilon" algorithm >>= number
    penalty <- field "penalty" algorithm >>= number
    count <- field "steps" algorithm >>= integer
    let ordered = zip [0 :: Natural ..] (order samples)
    (declared, entries) <- if chosen == Algorithm.grpo then grouped samples payload algorithm ordered else distilled samples payload ordered
    unless (count > 0 && count <= toInteger (length entries)) (Left (Shape "Optimizer steps must be between one and the number of samples"))
    let encoded = object (["specification" .= chosen, "policy" .= policy, "learner" .= checkpoint, "tokenizer" .= tokenizer, "base" .= base, "assembly" .= assembly, "behavior_model" .= behavior, "schedule" .= schedule, "reference" .= reference, "reference_source" .= source, "optimizer" .= optimizer, "epsilon" .= epsilon, "penalty" .= penalty, "steps" .= batches count (map label [0 .. length entries - 1]), "samples" .= entries, "order" .= map label [0 .. length entries - 1]] ++ declared)
    Request.value <$> first Shape (parseEither Request.parse encoded)

grouped :: Inputs -> V.Value Natural -> V.Value Natural -> [(Natural, Natural)] -> Either Error ([Pair], [Value])
grouped samples payload algorithm ordered = do
    rewards <- field "rewards" payload >>= mapping
    unless (Map.keysSet rewards == Map.keysSet (trajectories samples)) (Left (Membership "Rewards must share the trajectory keys"))
    grouping <- field "groups" payload >>= sequenceValues >>= traverse selectors
    unless (all ((>= minimumGroup) . length) grouping) (Left (Membership "Every advantage group must contain at least two samples"))
    let assignments = [(key, "g" ++ show index) | (index, group) <- zip [0 :: Natural ..] grouping, key <- group]
        groups = Map.fromList assignments
    unless (length assignments == Map.size (trajectories samples) && Map.keysSet groups == Map.keysSet (trajectories samples)) (Left (Membership "Groups must partition the trajectory keys"))
    delta <- field "delta" algorithm >>= number
    supplied <- traverse (\(position, key) -> Advantage.Reward (label position) <$> lookupKey key groups <*> (lookupKey key rewards >>= number)) ordered
    expected <- first Numerical (Advantage.calculate delta supplied)
    entries <- traverse (entry groups rewards expected) ordered
    pure (["delta" .= delta], entries)
  where
    entry groups rewards expected (position, key) = do
        shared <- sample samples (position, key)
        group <- lookupKey key groups
        reward <- lookupKey key rewards >>= number
        advantage <- maybe (Left (Membership "Missing expected sample advantage")) Right (Map.lookup (label position) expected)
        pure (object (shared ++ ["group" .= group, "reward" .= reward, "advantage_bits" .= advantage]))

distilled :: Inputs -> V.Value Natural -> [(Natural, Natural)] -> Either Error ([Pair], [Value])
distilled samples payload ordered = do
    teacher <- field "teacher" payload >>= teacherValue
    scored <- field "teacher_scores" payload >>= mapping
    calls <- field "teacher_calls" payload >>= mapping
    unless (Map.keysSet scored == Map.keysSet (trajectories samples) && Map.keysSet calls == Map.keysSet (trajectories samples)) (Left (Membership "Teacher scores and calls must share the trajectory keys"))
    entries <- traverse (entry scored calls) ordered
    pure (["teacher" .= teacher], entries)
  where
    entry scored calls (position, key) = do
        shared <- sample samples (position, key)
        words32 <- lookupKey key scored >>= sequenceValues >>= traverse bits
        call <- lookupKey key calls >>= bindingValue
        pure (object (shared ++ ["teacher_bits" .= words32, "teacher_call" .= call]))

image :: E.Emission -> Either Error Load.Image
image command = do
    (_, payload) <- emitted command
    learner <- field "learner" payload
    policy <- field "policy" learner >>= text
    checkpoint <- field "learner" learner >>= text
    tokenizer <- field "tokenizer" learner >>= text
    base <- field "base" learner >>= text
    assembly <- field "assembly" learner >>= text
    reference <- field "reference" payload >>= text
    actual <- field "policy" payload
    let expected = Materialization.learning (policy, checkpoint, tokenizer, base, assembly, reference)
    unless (actual == Load.imageValue expected) (Left (Shape "Learning load image differs from its actual input identities"))
    pure expected

emitted :: E.Emission -> Either Error (String, V.Value Natural)
emitted (E.Emission name chosen payload)
    | name == Algorithm.sink && chosen `elem` [Algorithm.grpo, Algorithm.opd] = Right (chosen, payload)
emitted _ = Left (Shape "Expected a learning update emission")

inputs :: V.Value Natural -> Either Error Inputs
inputs payload = do
    trajectories <- field "trajectories" payload >>= mapping
    generations <- field "generations" payload >>= mapping
    behavior <- field "behavior" payload >>= mapping
    scores <- field "reference_scores" payload >>= mapping
    order <- field "order" payload >>= sequenceValues >>= traverse singleton
    let keys = Map.keysSet trajectories
    unless (not (null order) && length order == Set.size keys && Set.fromList order == keys) (Left (Membership "Order must select every trajectory exactly once"))
    unless (Map.keysSet generations == keys && Map.keysSet behavior == keys && Map.keysSet scores == keys) (Left (Membership "Generations, behavior and reference scores must share the trajectory keys"))
    pure Inputs {trajectories, generations, behavior, scores, order}

minimumGroup :: Int
minimumGroup = 2

sample :: Inputs -> (Natural, Natural) -> Either Error [Pair]
sample batch (position, key) = do
    trajectory <- lookupKey key (trajectories batch)
    generation <- lookupKey key (generations batch)
    version <- field "version" generation >>= natural
    behaviorPolicy <- field "policy" generation >>= text
    probabilities <- lookupKey key (behavior batch) >>= sequenceValues >>= traverse bits
    scored <- lookupKey key (scores batch) >>= sequenceValues >>= traverse bits
    prompt <- field "prompt" trajectory >>= text
    seed <- field "seed" trajectory >>= integer
    limit <- field "limit" trajectory >>= natural
    temperature <- field "temperature" trajectory >>= number
    tokens <- field "tokens" trajectory >>= sequenceValues >>= traverse natural
    prefix <- field "prompt_length" trajectory >>= natural
    response <- field "text" trajectory >>= text
    truncated <- field "truncated" trajectory >>= boolean
    pure ["sample" .= label position, "prompt" .= prompt, "seed" .= seed, "limit" .= limit, "temperature" .= temperature, "tokens" .= tokens, "prompt_length" .= prefix, "version" .= version, "behavior_policy" .= behaviorPolicy, "behavior_bits" .= probabilities, "reference_bits" .= scored, "text" .= response, "truncated" .= truncated]

teacherValue :: V.Value Natural -> Either Error Value
teacherValue value = do
    named <- traverse (\name -> (Key.fromString name .=) <$> (field name value >>= text)) ["model", "revision", "adapter", "tokenizer", "base", "assembly", "truncation", "logprobs", "path"]
    temperature <- field "temperature" value >>= number
    pure (object (("temperature" .= temperature) : named))

bindingValue :: V.Value Natural -> Either Error Value
bindingValue value = do
    named <- traverse (\name -> (Key.fromString name .=) <$> (field name value >>= natural)) ["call", "attempt", "instance"]
    pure (object named)

modelValue :: V.Value Natural -> Either Error Value
modelValue value = do
    base <- field "base" value >>= text
    assembly <- field "assembly" value >>= text
    pure (object ["base" .= base, "assembly" .= assembly])

scheduleValue :: V.Value Natural -> Either Error Value
scheduleValue value = do
    update <- field "update" value >>= natural
    staleness <- field "staleness" value >>= natural
    pure (object ["update" .= update, "staleness" .= staleness])

optimizerValue :: V.Value Natural -> Either Error Value
optimizerValue value = do
    rate <- field "learning_rate" value >>= number
    betas <- field "betas" value >>= sequenceValues >>= traverse number
    epsilon <- field "epsilon" value >>= number
    decay <- field "weight_decay" value >>= number
    pure (object ["learning_rate" .= rate, "betas" .= betas, "epsilon" .= epsilon, "weight_decay" .= decay])

batches :: Integer -> [String] -> [[String]]
batches count = go (fromIntegral count)
  where
    go :: Int -> [String] -> [[String]]
    go 0 _ = []
    go remaining rest = let size = (length rest + remaining - 1) `div` remaining in take size rest : go (remaining - 1) (drop size rest)

label :: (Show value) => value -> String
label value = "s" ++ show value

field :: String -> V.Value Natural -> Either Error (V.Value Natural)
field name (V.Record fields) = maybe (Left (Shape ("Missing field: " ++ name))) Right (Map.lookup name fields)
field _ _ = Left (Shape "Expected a record")

mapping :: V.Value Natural -> Either Error (Map Natural (V.Value Natural))
mapping (V.Mapping values) = Right values
mapping _ = Left (Shape "Expected a keyed map")

sequenceValues :: V.Value Natural -> Either Error [V.Value Natural]
sequenceValues (V.Sequence values) = Right values
sequenceValues _ = Left (Shape "Expected a sequence")

selectors :: V.Value Natural -> Either Error [Natural]
selectors value = do
    entries <- mapping value
    unless (all (== V.Atom (V.Boolean True)) entries) (Left (Membership "A selector must contain only membership markers"))
    pure (Map.keys entries)

singleton :: V.Value Natural -> Either Error Natural
singleton value = do
    keys <- selectors value
    case keys of
        [key] -> Right key
        _ -> Left (Membership "An order position must name exactly one sample")

lookupKey :: Natural -> Map Natural value -> Either Error value
lookupKey key = maybe (Left (Membership "Missing linked sample")) Right . Map.lookup key

text :: V.Value Natural -> Either Error String
text value = sequenceValues value >>= traverse character
  where
    character (V.Atom (V.Token code))
        | code <= fromIntegral (ord (maxBound :: Char)) && not (code >= surrogateStart && code <= surrogateEnd) = Right (chr (fromIntegral code))
    character _ = Left (Shape "Expected a Unicode scalar value")
    surrogateStart = 0xd800
    surrogateEnd = 0xdfff

number :: V.Value Natural -> Either Error Double
number (V.Atom (V.Number value))
    | let result = fromRational value :: Double, not (isNaN result || isInfinite result) = first Shape (eitherDecode (encode (toJSON result)))
number _ = Left (Shape "Expected a finite numeric value")

integer :: V.Value Natural -> Either Error Integer
integer (V.Atom (V.Number value)) | denominator value == 1 = Right (numerator value)
integer _ = Left (Shape "Expected an integral value")

natural :: V.Value Natural -> Either Error Natural
natural (V.Atom (V.Token value)) = Right value
natural _ = Left (Shape "Expected a natural number")

bits :: V.Value Natural -> Either Error Value
bits (V.Atom (V.Bits32 value)) = Right (toJSON value)
bits _ = Left (Shape "Expected actual FP32 probability words")

boolean :: V.Value Natural -> Either Error Bool
boolean (V.Atom (V.Boolean value)) = Right value
boolean _ = Left (Shape "Expected a Boolean")
