{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Request (Request, parse, value, logical, exchange, order, materialization, scheduled, staleness, version) where

import Control.Monad (unless, when, (>=>))
import Data.Aeson (Key, Object, Value (Object), parseJSON, toJSON, withObject, (.:))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Word (Word32)
import Invar.Float32 qualified as Float32
import Invar.Json qualified as Json
import Invar.Learn.Algorithm qualified as Algorithm
import Invar.Learn.Objective qualified as Objective
import Invar.Learn.Stream qualified as S
import Invar.Materialization qualified as Materialization
import Invar.Spec.Load qualified as Image
import Numeric.Natural (Natural)

data Request = Request Value Value S.Plan [Text] Image.Image (Natural, Natural, Natural)
    deriving (Eq, Show)

value :: Request -> Value
value (Request original _ _ _ _ _) = original

logical :: Request -> Value
logical (Request _ ordered _ _ _ _) = ordered

exchange :: Request -> S.Plan
exchange (Request _ _ planned _ _ _) = planned

order :: Request -> [Text]
order (Request _ _ _ sequence' _ _) = sequence'

materialization :: Request -> Image.Image
materialization (Request _ _ _ _ image _) = image

scheduled :: Request -> Natural
scheduled (Request _ _ _ _ _ (update, _, _)) = update

staleness :: Request -> Natural
staleness (Request _ _ _ _ _ (_, lag, _)) = lag

version :: Request -> Natural
version (Request _ _ _ _ _ (_, _, behavior)) = behavior

parse :: Value -> Parser Request
parse = withObject "numerical update request" $ \fields -> do
    specification <- fields .: "specification"
    if specification == Text.pack Algorithm.grpo then grouped fields else if specification == Text.pack Algorithm.opd then distilled fields else fail "Unsupported update specification"

grouped :: Object -> Parser Request
grouped fields = do
    Json.fields (common ++ ["delta"]) fields
    raw <- fields .: "samples" :: Parser [Value]
    groups <- traverse (withObject "update sample" (.: "group")) raw :: Parser [Text]
    when (any Text.null groups) (fail "Logical sample and group identities must be nonempty")
    unless (all (>= minimumGroup) (Map.fromListWith (+) [(group, 1 :: Int) | group <- groups])) (fail "Each advantage group requires at least two samples")
    epsilon <- fields .: "epsilon" >>= Json.finite
    penalty <- fields .: "penalty" >>= Json.finite
    delta <- fields .: "delta" >>= Json.finite
    unless (Algorithm.coefficients epsilon penalty delta) (fail "Invalid GRPO coefficient configuration")
    shared fields (S.Grouped, Objective.Profile epsilon penalty) (["group", "reward", "advantage_bits"], advantage)
  where
    advantage sampled _ = do
        _ <- sampled .: "reward" >>= Json.finite
        S.Advantage <$> (sampled .: "advantage_bits" >>= word False)

distilled :: Object -> Parser Request
distilled fields = do
    Json.fields (common ++ ["teacher"]) fields
    teacher <- fields .: "teacher" >>= teacherIdentity
    epsilon <- fields .: "epsilon" >>= Json.finite
    penalty <- fields .: "penalty" >>= Json.finite
    unless (Algorithm.clipping epsilon && Algorithm.penalty penalty) (fail "Invalid distillation coefficient configuration")
    shared fields (S.Distilled teacher, Objective.Profile epsilon penalty) (["teacher_bits", "teacher_call"], scored)
  where
    scored sampled count = do
        words32 <- (sampled .: "teacher_bits" :: Parser [Value]) >>= traverse (word True)
        unless (length words32 == count) (fail "Teacher scores must cover every response token")
        sampled .: "teacher_call" >>= call
        pure (S.Teacher words32)

common :: [Key]
common = ["specification", "policy", "learner", "reference", "reference_source", "tokenizer", "base", "assembly", "behavior_model", "schedule", "samples", "order", "steps", "epsilon", "penalty", "optimizer"]

shared :: Object -> (S.Algorithm, Objective.Profile) -> ([Key], Object -> Int -> Parser S.Signal) -> Parser Request
shared fields (kind, profile) extension = do
    source <- fields .: "reference_source" >>= maybe (fail "Unknown reference source") pure . S.readSource
    [policy, checkpoint, reference, tokenizer, base, assembly] <- traverse ((.:) fields >=> Json.identity) ["policy", "learner", "reference", "tokenizer", "base", "assembly"]
    fields .: "behavior_model" >>= model
    (update, lag) <- fields .: "schedule" >>= schedule
    delivered <- fields .: "samples" >>= traverse (sample extension)
    sequence' <- fields .: "order" :: Parser [Text]
    let behavior = if update > lag then update - lag else 0
    unless (all (\entry -> sampleVersion entry == behavior) delivered) (fail "Every sample must come from version max(0, update - staleness)")
    unless (Set.size (Set.fromList (map samplePolicy delivered)) == 1) (fail "Every sample of one version must come from the one policy published as that version")
    unless (behavior /= update || all (\entry -> samplePolicy entry == Text.pack policy) delivered) (fail "Samples of the update's own version must come from the policy being updated")
    unless (all (\entry -> not (null (S.referenceWords (sampleWords entry))) == (samplePolicy entry /= Text.pack reference)) delivered) (fail "Reference scores must be present exactly when the reference differs from the sample's behavior policy")
    let named = Map.fromList [(sampleName entry, sampleValue entry) | entry <- delivered]
    unless (not (null delivered) && Map.size named == length delivered) (fail "Cohort samples must be nonempty and distinct")
    unless (length sequence' == Map.size named && Set.fromList sequence' == Map.keysSet named) (fail "Logical order must name every admitted sample exactly once")
    ordered <- traverse (maybe (fail "Missing logical sample") pure . (`Map.lookup` named)) sequence'
    steps <- fields .: "steps" :: Parser [[Text]]
    when (null steps || any null steps) (fail "An update needs nonempty optimizer steps")
    unless (Set.fromList (concat steps) == Map.keysSet named) (fail "Optimizer steps must use every admitted sample and no other")
    fields .: "optimizer" >>= optimizer
    let planned = S.Plan profile policy (map sampleWords delivered) steps source kind
        image = Materialization.learning (policy, checkpoint, tokenizer, base, assembly, reference)
    pure (Request (Object fields) (Object (Fields.insert "samples" (toJSON ordered) fields)) planned sequence' image (update, lag, behavior))

teacherIdentity :: Value -> Parser Algorithm.Teacher
teacherIdentity = withObject "teacher identity" $ \fields -> do
    Json.fields ["model", "revision", "adapter", "tokenizer", "base", "assembly", "temperature", "truncation", "logprobs", "path"] fields
    [model', revision, adapter, declared, base, assembly] <- traverse (fields .:) ["model", "revision", "adapter", "tokenizer", "base", "assembly"]
    temperature <- fields .: "temperature" >>= Json.finite
    [truncation, logprobs, path] <- traverse (fields .:) ["truncation", "logprobs", "path"]
    let chosen = Algorithm.Teacher model' revision adapter declared base assembly temperature truncation logprobs path
    either fail (const (pure chosen)) (Algorithm.teacher chosen)

call :: Value -> Parser ()
call = withObject "teacher scoring call" $ \fields -> do
    Json.fields ["call", "attempt", "instance"] fields
    mapM_ (\name -> fields .: name :: Parser Natural) ["call", "attempt", "instance"]

schedule :: Value -> Parser (Natural, Natural)
schedule = withObject "update schedule" $ \fields -> do
    Json.fields ["update", "staleness"] fields
    (,) <$> fields .: "update" <*> fields .: "staleness"

data Delivered = Delivered {sampleName :: Text, sampleValue :: Value, sampleVersion :: Natural, samplePolicy :: Text, sampleWords :: S.Sample}

model :: Value -> Parser ()
model = withObject "behavior model representation" $ \fields -> do
    Json.fields ["base", "assembly"] fields
    mapM_ ((.:) fields >=> Json.identity) ["base", "assembly"]

minimumGroup :: Int
minimumGroup = 2

sample :: ([Key], Object -> Int -> Parser S.Signal) -> Value -> Parser Delivered
sample (extra, signalOf) original = withObject "update sample" inspect original
  where
    inspect fields = do
        Json.fields (["sample", "prompt", "seed", "limit", "temperature", "tokens", "prompt_length", "version", "behavior_policy", "behavior_bits", "reference_bits", "text", "truncated"] ++ extra) fields
        named <- fields .: "sample"
        when (Text.null named) (fail "Logical sample identities must be nonempty")
        sampled <- fields .: "version"
        behaviorPolicy <- fields .: "behavior_policy" >>= Json.identity
        _ <- fields .: "prompt" :: Parser Text
        _ <- fields .: "text" :: Parser Text
        _ <- fields .: "seed" :: Parser Integer
        limit <- fields .: "limit" :: Parser Natural
        temperature <- fields .: "temperature" >>= Json.finite
        unless (limit > 0 && temperature > 0) (fail "Sample limit and temperature must be positive")
        tokens <- fields .: "tokens" :: Parser [Natural]
        prefix <- fields .: "prompt_length" :: Parser Natural
        words32 <- (fields .: "behavior_bits" :: Parser [Value]) >>= traverse (word True)
        let count = fromIntegral (length words32)
        unless (prefix > 0 && prefix < fromIntegral (length tokens) && count == fromIntegral (length tokens) - prefix && count <= limit) (fail "Behavior observations must match the admitted response tokens")
        scores <- (fields .: "reference_bits" :: Parser [Value]) >>= traverse (word True)
        unless (null scores || length scores == length words32) (fail "Reference scores must cover every response token")
        truncated <- fields .: "truncated"
        unless (not truncated || count == limit) (fail "Invalid observed truncation status")
        learned <- signalOf fields (length words32)
        pure (Delivered named original sampled (Text.pack behaviorPolicy) (S.Sample named words32 scores learned))

word :: Bool -> Value -> Parser Word32
word probability encoded = do
    decoded <- parseJSON encoded
    unless (if probability then Float32.logProbability decoded else Float32.finite decoded) (fail "Expected finite FP32 words and nonpositive log probabilities")
    pure decoded

optimizer :: Value -> Parser ()
optimizer = withObject "AdamW configuration" $ \fields -> do
    Json.fields ["learning_rate", "betas", "epsilon", "weight_decay"] fields
    rate <- fields .: "learning_rate" >>= Json.finite
    betas <- (fields .: "betas" :: Parser [Value]) >>= traverse Json.finite
    epsilon <- fields .: "epsilon" >>= Json.finite
    decay <- fields .: "weight_decay" >>= Json.finite
    case betas of
        [first', second'] | Algorithm.optimizer (Algorithm.Optimizer rate first' second' epsilon decay) -> pure ()
        _ -> fail "Invalid AdamW scalar configuration"
