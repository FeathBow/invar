{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Request (Request, parse, value, logical, exchange, order, materialization) where

import Control.Monad (unless, when, (>=>))
import Data.Aeson (Value (Object), parseJSON, toJSON, withObject, (.:))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Word (Word32)
import Invar.Float32 qualified as Float32
import Invar.Json qualified as Json
import Invar.Learn.Objective qualified as Objective
import Invar.Learn.Stream qualified as S
import Invar.Materialization qualified as Materialization
import Invar.Spec.Load qualified as Image
import Numeric.Natural (Natural)

data Request = Request Value Value S.Plan [Text] Image.Image
    deriving (Eq, Show)

value :: Request -> Value
value (Request original _ _ _ _) = original

logical :: Request -> Value
logical (Request _ ordered _ _ _) = ordered

exchange :: Request -> S.Plan
exchange (Request _ _ planned _ _) = planned

order :: Request -> [Text]
order (Request _ _ _ sequence' _) = sequence'

materialization :: Request -> Image.Image
materialization (Request _ _ _ _ image) = image

parse :: Value -> Parser Request
parse = withObject "numerical update request" $ \fields -> do
    Json.fields ["specification", "policy", "learner", "reference", "reference_source", "tokenizer", "base", "assembly", "behavior_model", "schedule", "samples", "order", "steps", "epsilon", "penalty", "delta", "optimizer"] fields
    specification <- fields .: "specification"
    unless (specification == ("grpo-token-mean/v1" :: Text)) (fail "Unsupported update specification")
    source <- fields .: "reference_source" >>= maybe (fail "Unknown reference source") pure . S.readSource
    [policy, checkpoint, reference, tokenizer, base, assembly] <- traverse ((.:) fields >=> Json.identity) ["policy", "learner", "reference", "tokenizer", "base", "assembly"]
    fields .: "behavior_model" >>= model
    (update, staleness) <- fields .: "schedule" >>= schedule
    delivered <- fields .: "samples" >>= traverse sample
    sequence' <- fields .: "order" :: Parser [Text]
    let version = if update > staleness then update - staleness else 0
    unless (all (\entry -> sampleVersion entry == version) delivered) (fail "Every sample must come from version max(0, update - staleness)")
    unless (Set.size (Set.fromList (map samplePolicy delivered)) == 1) (fail "Every sample of one version must come from the one policy published as that version")
    unless (version /= update || all (\entry -> samplePolicy entry == Text.pack policy) delivered) (fail "Samples of the update's own version must come from the policy being updated")
    unless (all (\entry -> not (null (S.referenceWords (sampleWords entry))) == (samplePolicy entry /= Text.pack reference)) delivered) (fail "Reference scores must be present exactly when the reference differs from the sample's behavior policy")
    let named = Map.fromList [(sampleName entry, sampleValue entry) | entry <- delivered]
        grouped = Map.fromListWith (+) [(sampleGroup entry, 1 :: Int) | entry <- delivered]
    unless (not (null delivered) && Map.size named == length delivered) (fail "Cohort samples must be nonempty and distinct")
    unless (length sequence' == Map.size named && Set.fromList sequence' == Map.keysSet named) (fail "Logical order must name every admitted sample exactly once")
    unless (all (>= minimumGroup) grouped) (fail "Each advantage group requires at least two samples")
    ordered <- traverse (maybe (fail "Missing logical sample") pure . (`Map.lookup` named)) sequence'
    steps <- fields .: "steps" :: Parser [[Text]]
    when (null steps || any null steps) (fail "An update needs nonempty optimizer steps")
    unless (Set.fromList (concat steps) == Map.keysSet named) (fail "Optimizer steps must use every admitted sample and no other")
    epsilon <- fields .: "epsilon" >>= Json.finite
    penalty <- fields .: "penalty" >>= Json.finite
    delta <- fields .: "delta" >>= Json.finite
    unless (epsilon > 0 && epsilon < 1 && penalty >= 0 && delta > 0) (fail "Invalid GRPO coefficient configuration")
    fields .: "optimizer" >>= optimizer
    let planned = S.Plan (Objective.Profile epsilon penalty) policy (map sampleWords delivered) steps source
        image = Materialization.learning (policy, checkpoint, tokenizer, base, assembly, reference)
    pure (Request (Object fields) (Object (Fields.insert "samples" (toJSON ordered) fields)) planned sequence' image)

schedule :: Value -> Parser (Natural, Natural)
schedule = withObject "update schedule" $ \fields -> do
    Json.fields ["update", "staleness"] fields
    (,) <$> fields .: "update" <*> fields .: "staleness"

data Delivered = Delivered {sampleName :: Text, sampleGroup :: Text, sampleValue :: Value, sampleVersion :: Natural, samplePolicy :: Text, sampleWords :: S.Sample}

model :: Value -> Parser ()
model = withObject "behavior model representation" $ \fields -> do
    Json.fields ["base", "assembly"] fields
    mapM_ ((.:) fields >=> Json.identity) ["base", "assembly"]

minimumGroup :: Int
minimumGroup = 2

sample :: Value -> Parser Delivered
sample original = withObject "update sample" inspect original
  where
    inspect fields = do
        Json.fields ["sample", "group", "prompt", "seed", "limit", "temperature", "tokens", "prompt_length", "version", "behavior_policy", "behavior_bits", "reference_bits", "text", "truncated", "reward", "advantage_bits"] fields
        named <- fields .: "sample"
        grouped <- fields .: "group"
        when (Text.null named || Text.null grouped) (fail "Logical sample and group identities must be nonempty")
        version <- fields .: "version"
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
        _ <- fields .: "reward" >>= Json.finite
        advantage <- fields .: "advantage_bits" >>= word False
        pure (Delivered named grouped original version (Text.pack behaviorPolicy) (S.Sample named words32 scores advantage))

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
    let moments = 2
    unless (rate >= 0 && epsilon > 0 && decay >= 0 && length betas == moments && all (\beta -> beta >= 0 && beta < 1) betas) (fail "Invalid AdamW scalar configuration")
