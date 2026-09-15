{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Checkpoint (Checked, Check (..), admit, admitInitial, value, tensors, inspectTensors, rngSummary) where

import Control.Monad (unless, when)
import Data.Aeson ((.:), (.=))
import Data.Aeson qualified as Json
import Data.Aeson.Key qualified as Key
import Data.Aeson.Types (Pair, Parser, parseEither)
import Data.Bifunctor (first)
import Data.ByteString qualified as Bytes
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (catMaybes)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import GHC.Float (castWord32ToFloat)
import Invar.Json qualified as Fields
import Invar.Learn qualified as Learn
import Invar.Learn.Adapter qualified as Adapter
import Invar.Learn.Checkpoint.MLX qualified as MLX
import Invar.Learn.Checkpoint.Types
import Invar.Learn.Codec qualified as Codec
import Invar.Learn.Native qualified as Native
import Invar.Learn.Report qualified as Report
import Invar.Policy.File qualified as File
import Invar.Policy.Header qualified as Header

data Setting = Number Rational | Boolean Bool | Betas [Rational]

value :: Checked -> Native.Value
value (Checked observed _ _ _) = observed

tensors :: Checked -> [(Check, Native.Tensor)]
tensors (Checked _ checks _ _) = checks

inspectTensors :: Codec.Session -> Checked -> IO [Integer]
inspectTensors session checked = do
    steps <- catMaybes <$> traverse inspect (tensors checked)
    let Checked _ _ random _ = checked
    mapM_ (\tensor -> Codec.consume (session, tensor) (const (pure ()))) random
    pure steps
  where
    inspect (Finite, tensor) = Codec.consume (session, tensor) finite >> pure Nothing
    inspect (Step, tensor) = do
        encoded <- Codec.chunk session tensor (0, Header.fp32Bytes)
        finite encoded
        let observed = castWord32ToFloat (fromInteger (unsigned encoded))
        unless (observed > 0 && fromInteger (truncate observed) == observed) (invalid "Expected a positive integral post-update AdamW step")
        pure (Just (truncate observed))
    inspect (IntegerStep expected, tensor) = do
        encoded <- Codec.chunk session tensor (0, Native.size tensor)
        let observed = unsigned encoded
        case expected of
            Initial -> unless (observed == 0) (invalid "Expected initial MLX AdamW step zero") >> pure Nothing
            Updated -> unless (observed > 0) (invalid "Expected positive post-update MLX AdamW step") >> pure (Just observed)
    inspect (Word32Equal expected, tensor) = do
        encoded <- Codec.chunk session tensor (0, Header.fp32Bytes)
        finite encoded
        unless (unsigned encoded == toInteger expected) (invalid "Actual MLX learning rate differs from the consumed FP32 value")
        pure Nothing
    unsigned = Bytes.foldr (\byte accumulated -> fromIntegral byte + octetBase * accumulated) 0
    octetBase = 256
    finite encoded = unless (File.finite encoded) (invalid "Expected finite FP32 AdamW state tensors")

rngSummary :: Checked -> [Pair]
rngSummary (Checked _ _ _ summary) = summary

admit :: Report.Report -> Map Text [Integer] -> Native.Value -> Either String Checked
admit report parameters observed = do
    expected <- Text.pack <$> Report.artifact "adapter" report
    input <- parseEither (Json.withObject "consumed numerical request" pure) (Report.request report)
    bindings <- traverse (\name -> (name,) <$> parseEither (.: Key.fromText name) input) ["base", "assembly", "tokenizer"]
    configuration <- parseEither (\request -> request .: "optimizer" >>= settings) input
    checkpoint (Expected Updated expected bindings configuration) parameters observed

admitInitial :: Learn.Settings -> Map Text [Integer] -> Native.Value -> Either String Checked
admitInitial requested parameters observed = do
    first show (Learn.validate requested)
    let chosen = Learn.optimizer requested
        bindings = [("base", Text.pack (Learn.base requested)), ("assembly", Text.pack (Learn.assembly requested)), ("tokenizer", Text.pack (Learn.tokenizer requested))]
        configuration = AdamW (Learn.learningRate chosen) [Learn.firstMoment chosen, Learn.secondMoment chosen] (Learn.epsilon chosen) (Learn.weightDecay chosen)
    checkpoint (Expected Initial (Text.pack (Learn.policy requested)) bindings configuration) parameters observed

checkpoint :: Expected -> Map Text [Integer] -> Native.Value -> Either String Checked
checkpoint expected parameters observed = do
    fields <- Native.mapping observed
    case Map.lookup "format" fields of
        Just (Native.String "invar-mlx-learner/v1") -> MLX.checkpoint expected parameters observed
        Nothing -> torchCheckpoint expected parameters observed
        _ -> Left "Unknown learner checkpoint format"

torchCheckpoint :: Expected -> Map Text [Integer] -> Native.Value -> Either String Checked
torchCheckpoint expected parameters observed = do
    when (Adapter.mlxParameters parameters) (Left "Torch checkpoint requires PEFT parameter bindings")
    fields <- exact ("learner checkpoint", ["adapter", "base", "assembly", "tokenizer", "parameters", "optimizer", "cpu_rng", "cuda_rng"]) observed
    adapter <- field "adapter" fields >>= Native.string
    unless (adapter == policy expected) (Left "Checkpoint adapter binding mismatch")
    mapM_ (materialization fields) (bindings expected)
    names <- field "parameters" fields >>= parameterNames parameters
    original <- field "optimizer" fields
    (optimizer, checks) <- optimizerState (phase expected, adamw (optimizer expected), parameters, names) original
    cpu <- field "cpu_rng" fields >>= rng
    cuda <- field "cuda_rng" fields >>= Native.list >>= traverse rng
    let canonical = Map.insert "parameters" (sequenceValue (map Native.String (Map.keys parameters))) (Map.insert "optimizer" optimizer fields)
    pure (Checked (mappingValue canonical) checks (cpu : cuda) ["cpu_rng_bytes" .= Native.size cpu, "cuda_rng_bytes" .= map Native.size cuda])

settings :: Json.Value -> Parser AdamW
settings = Json.withObject "consumed AdamW configuration" $ \fields -> do
    rate <- fields .: "learning_rate" >>= Fields.finite
    coefficients <- (fields .: "betas" :: Parser [Json.Value]) >>= traverse Fields.finite
    epsilon <- fields .: "epsilon" >>= Fields.finite
    decay <- fields .: "weight_decay" >>= Fields.finite
    pure (AdamW rate coefficients epsilon decay)

adamw :: AdamW -> Map Text Setting
adamw (AdamW rate coefficients epsilon decay) = Map.fromList ([("lr", Number (toRational rate)), ("betas", Betas (map toRational coefficients)), ("eps", Number (toRational epsilon)), ("weight_decay", Number (toRational decay)), ("decoupled_weight_decay", Boolean True)] ++ disabled)
  where
    disabled = [(name, Boolean False) | name <- ["foreach", "fused", "amsgrad", "maximize", "capturable", "differentiable"]]

parameterNames :: Map Text [Integer] -> Native.Value -> Either String (Map Integer Text)
parameterNames expected encoded = do
    names <- Native.indexed encoded >>= traverse Native.string
    unless (Map.size names == Map.size expected && Set.fromList (Map.elems names) == Map.keysSet expected) (Left "Optimizer parameter binding inventory mismatch")
    pure names

optimizerState :: (Phase, Map Text Setting, Map Text [Integer], Map Integer Text) -> Native.Value -> Either String (Native.Value, [(Check, Native.Tensor)])
optimizerState (phase, configuration, parameters, names) original = do
    optimizer <- exact ("optimizer checkpoint", ["state", "param_groups"]) original
    group <- field "param_groups" optimizer >>= Native.list >>= singleGroup
    groupFields <- exact ("AdamW parameter group", "params" : Map.keys configuration) group
    mapM_ (\(name, setting) -> field name groupFields >>= matches setting) (Map.toAscList configuration)
    identities <- field "params" groupFields >>= Native.list >>= traverse Native.integer
    unless (all (>= 0) identities && length identities == Map.size names && Set.fromList identities == Map.keysSet names) (Left "Optimizer parameter inventory mismatch")
    slots <- field "state" optimizer >>= Native.indexed
    checked <- case phase of
        Initial -> do
            unless (Map.null slots) (Left "Expected empty initial AdamW state")
            pure []
        Updated -> do
            unless (Map.keysSet slots == Map.keysSet names) (Left "Optimizer slot inventory mismatch")
            traverse (namedSlot (parameters, slots)) (Map.toAscList names)
    let canonicalSlots = Map.fromList [(name, state) | (name, state, _) <- checked]
        canonicalGroup = mappingValue (Map.insert "params" (sequenceValue (map Native.String (Map.keys parameters))) groupFields)
        canonical = mappingValue (Map.insert "state" (mappingValue canonicalSlots) (Map.insert "param_groups" (sequenceValue [canonicalGroup]) optimizer))
    pure (canonical, concatMap (\(_, _, checks) -> checks) checked)
  where
    singleGroup [group] = Right group
    singleGroup _ = Left "Expected the single-group AdamW profile"

matches :: Setting -> Native.Value -> Either String ()
matches (Boolean expected) (Native.Boolean actual) = unless (actual == expected) mismatch
matches (Number expected) actual = Native.number actual >>= \number -> unless (number == expected) mismatch
matches (Betas expected) actual = do
    coefficients <- Native.tuple actual >>= traverse Native.number
    unless (coefficients == expected) mismatch
matches _ _ = mismatch

mismatch :: Either String value
mismatch = Left "AdamW settings differ from the consumed input"

namedSlot :: (Map Text [Integer], Map Integer Native.Value) -> (Integer, Text) -> Either String (Text, Native.Value, [(Check, Native.Tensor)])
namedSlot (parameters, slots) (identity, name) = do
    original <- maybe (Left "Missing named optimizer state") Right (Map.lookup identity slots)
    shape <- maybe (Left "Missing named adapter parameter") Right (Map.lookup name parameters)
    checks <- slot shape original
    pure (name, original, checks)

slot :: [Integer] -> Native.Value -> Either String [(Check, Native.Tensor)]
slot expected original = do
    fields <- exact ("AdamW slot", ["step", "exp_avg", "exp_avg_sq"]) original
    step <- field "step" fields >>= fp32
    average <- field "exp_avg" fields >>= fp32
    square <- field "exp_avg_sq" fields >>= fp32
    unless (null (Native.shape step)) (Left "Expected a positive integral post-update AdamW step")
    unless (Native.shape average == Native.shape square && product (Native.shape average) > 0) (Left "AdamW moment shapes disagree")
    unless (Native.shape average == expected) (Left "AdamW moment inventory parameter shape mismatch")
    pure [(Step, step), (Finite, average), (Finite, square)]

fp32 :: Native.Value -> Either String Native.Tensor
fp32 (Native.Tensor tensor) = do
    unless (Native.dtype tensor == "torch.float32" && Native.layout tensor == "torch.strided" && Native.size tensor == Header.fp32Bytes * product (Native.shape tensor)) (Left "Expected finite FP32 AdamW state tensors")
    pure tensor
fp32 _ = Left "Expected finite FP32 AdamW state tensors"

rng :: Native.Value -> Either String Native.Tensor
rng (Native.Tensor tensor) = do
    unless (Native.dtype tensor == "torch.uint8" && Native.layout tensor == "torch.strided" && length (Native.shape tensor) == 1 && Native.size tensor > 0 && Native.size tensor == product (Native.shape tensor)) (Left "Expected a nonempty RNG byte vector")
    pure tensor
rng _ = Left "Expected a nonempty RNG byte vector"

invalid :: String -> IO value
invalid = ioError . userError
