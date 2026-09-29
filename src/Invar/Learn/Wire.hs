{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Wire (Error (..), lower, image) where

import Control.Monad (unless)
import Data.Aeson (Value, eitherDecode, encode, object, toJSON, (.=))
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.Char (chr, ord)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Ratio (denominator, numerator)
import Data.Set qualified as Set
import Data.Word (Word32)
import Invar.Learn.Advantage qualified as Advantage
import Invar.Learn.Request qualified as Request
import Invar.Materialization qualified as Materialization
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Load qualified as Load
import Invar.Spec.Value qualified as V
import Numeric.Natural (Natural)

data Error = Shape String | Membership String | Numerical Advantage.Error
    deriving (Eq, Show)

data Inputs = Inputs {trajectories :: Map Natural (V.Value Natural), behavior :: Map Natural (V.Value Natural), scores :: Map Natural (V.Value Natural), rewards :: Map Natural (V.Value Natural), groups :: Map Natural String, order :: [Natural]}

lower :: E.Emission -> Either Error Value
lower command@(E.Emission "update" "grpo-token-mean/v1" payload) = do
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
    reference <- field "reference" payload >>= text
    optimizer <- field "optimizer" learner >>= optimizerValue
    epsilon <- field "epsilon" algorithm >>= number
    penalty <- field "penalty" algorithm >>= number
    delta <- field "delta" algorithm >>= number
    count <- field "steps" algorithm >>= integer
    let ordered = zip [0 :: Natural ..] (order samples)
    supplied <- traverse (rewardInput samples) ordered
    expected <- first Numerical (Advantage.calculate delta supplied)
    entries <- traverse (sample samples expected) ordered
    unless (count > 0 && count <= toInteger (length entries)) (Left (Shape "Optimizer steps must be between one and the number of samples"))
    let encoded = object ["specification" .= ("grpo-token-mean/v1" :: String), "policy" .= policy, "learner" .= checkpoint, "tokenizer" .= tokenizer, "base" .= base, "assembly" .= assembly, "behavior_model" .= behavior, "reference" .= reference, "optimizer" .= optimizer, "epsilon" .= epsilon, "penalty" .= penalty, "delta" .= delta, "steps" .= batches count (map label [0 .. length entries - 1]), "samples" .= entries, "order" .= map label [0 .. length entries - 1]]
    Request.value <$> first Shape (parseEither Request.parse encoded)
lower _ = Left (Shape "Expected the GRPO update emission")

image :: E.Emission -> Either Error Load.Image
image (E.Emission "update" "grpo-token-mean/v1" payload) = do
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
image _ = Left (Shape "Expected the GRPO update emission")

inputs :: V.Value Natural -> Either Error Inputs
inputs payload = do
    trajectories <- field "trajectories" payload >>= mapping
    behavior <- field "behavior" payload >>= mapping
    scores <- field "reference_scores" payload >>= mapping
    rewards <- field "rewards" payload >>= mapping
    order <- field "order" payload >>= sequenceValues >>= traverse singleton
    let keys = Map.keysSet trajectories
    unless (not (null order) && length order == Set.size keys && Set.fromList order == keys) (Left (Membership "Order must select every trajectory exactly once"))
    unless (Map.keysSet behavior == keys && Map.keysSet scores == keys && Map.keysSet rewards == keys) (Left (Membership "Behavior, reference scores and rewards must share the trajectory keys"))
    grouped <- field "groups" payload >>= sequenceValues >>= traverse selectors
    unless (all ((>= minimumGroup) . length) grouped) (Left (Membership "Every advantage group must contain at least two samples"))
    let assignments = [(key, "g" ++ show index) | (index, group) <- zip [0 :: Natural ..] grouped, key <- group]
        groups = Map.fromList assignments
    unless (length assignments == Set.size keys && Map.keysSet groups == keys) (Left (Membership "Groups must partition the trajectory keys"))
    pure Inputs {trajectories, behavior, scores, rewards, groups, order}

minimumGroup :: Int
minimumGroup = 2

rewardInput :: Inputs -> (Natural, Natural) -> Either Error Advantage.Reward
rewardInput batch (position, key) = do
    assigned <- lookupKey key (groups batch)
    value <- lookupKey key (rewards batch) >>= number
    pure Advantage.Reward {Advantage.sample = label position, Advantage.group = assigned, Advantage.value = value}

sample :: Inputs -> Map String Word32 -> (Natural, Natural) -> Either Error Value
sample batch expected (position, key) = do
    trajectory <- lookupKey key (trajectories batch)
    group <- lookupKey key (groups batch)
    probabilities <- lookupKey key (behavior batch) >>= sequenceValues >>= traverse bits
    scored <- lookupKey key (scores batch) >>= sequenceValues >>= traverse bits
    reward <- lookupKey key (rewards batch) >>= number
    advantage <- maybe (Left (Membership "Missing expected sample advantage")) Right (Map.lookup (label position) expected)
    prompt <- field "prompt" trajectory >>= text
    seed <- field "seed" trajectory >>= integer
    limit <- field "limit" trajectory >>= natural
    temperature <- field "temperature" trajectory >>= number
    tokens <- field "tokens" trajectory >>= sequenceValues >>= traverse natural
    prefix <- field "prompt_length" trajectory >>= natural
    response <- field "text" trajectory >>= text
    truncated <- field "truncated" trajectory >>= boolean
    pure (object ["sample" .= label position, "group" .= group, "prompt" .= prompt, "seed" .= seed, "limit" .= limit, "temperature" .= temperature, "tokens" .= tokens, "prompt_length" .= prefix, "behavior_bits" .= probabilities, "reference_bits" .= scored, "text" .= response, "truncated" .= truncated, "reward" .= reward, "advantage_bits" .= advantage])

modelValue :: V.Value Natural -> Either Error Value
modelValue value = do
    base <- field "base" value >>= text
    assembly <- field "assembly" value >>= text
    pure (object ["base" .= base, "assembly" .= assembly])

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
