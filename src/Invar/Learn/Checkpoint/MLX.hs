{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Checkpoint.MLX (checkpoint) where

import Control.Monad (unless)
import Data.Aeson ((.=))
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import GHC.Float (castFloatToWord32, double2Float)
import Invar.Learn.Adapter qualified as Adapter
import Invar.Learn.Checkpoint.Types
import Invar.Learn.Native qualified as Native
import Invar.Policy.Header qualified as Header

checkpoint :: Expected -> Map Text [Integer] -> Native.Value -> Either String Checked
checkpoint expected parameters observed = do
    unless (Adapter.mlxParameters parameters) (Left "MLX checkpoint requires native LoRA parameters")
    fields <- exact ("MLX learner checkpoint", ["format", "adapter", "base", "assembly", "tokenizer", "parameters", "optimizer", "rng"]) observed
    format <- field "format" fields >>= Native.string
    unless (format == "invar-mlx-learner/v1") (Left "Unknown MLX learner checkpoint format")
    adapter <- field "adapter" fields >>= Native.string
    unless (adapter == policy expected) (Left "Checkpoint adapter binding mismatch")
    mapM_ (materialization fields) (bindings expected)
    names <- field "parameters" fields >>= Native.list >>= traverse Native.string
    unless (length names == Map.size parameters && Set.fromList names == Map.keysSet parameters) (Left "Optimizer parameter binding inventory mismatch")
    checks <- field "optimizer" fields >>= optimizerState expected parameters
    random <- field "rng" fields >>= Native.list >>= traverse rng
    unless (length random == 1) (Left "Expected one native MLX PRNG key")
    let canonical = Map.insert "parameters" (sequenceValue (map Native.String (Map.keys parameters))) fields
    pure (Checked (mappingValue canonical) checks random ["mlx_rng_bytes" .= map Native.size random])

optimizerState :: Expected -> Map Text [Integer] -> Native.Value -> Either String [(Check, Native.Tensor)]
optimizerState expected parameters observed = do
    fields <- exact ("MLX AdamW checkpoint", ["configuration", "state"]) observed
    field "configuration" fields >>= configuration (optimizer expected)
    state <- field "state" fields >>= Native.mapping
    let moments = Map.fromList [(name <> suffix, shape) | (name, shape) <- Map.toAscList parameters, suffix <- [".m", ".v"]]
        expectedMoments = case phase expected of Initial -> Map.empty; Updated -> moments
    unless (Map.keysSet state == Set.fromList ["step", "learning_rate"] `Set.union` Map.keysSet expectedMoments) (Left "MLX AdamW slot inventory mismatch")
    step <- field "step" state >>= tensor ("mlx.core.uint64", [], uint64Bytes)
    learningRate <- field "learning_rate" state >>= tensor ("mlx.core.float32", [], Header.fp32Bytes)
    checked <- traverse (moment state) (Map.toAscList expectedMoments)
    pure ((IntegerStep (phase expected), step) : (Word32Equal (castFloatToWord32 (double2Float (rate (optimizer expected)))), learningRate) : checked)
  where
    moment state (name, shape) = do
        actual <- field name state >>= tensor ("mlx.core.float32", shape, Header.fp32Bytes)
        unless (not (null shape) && product shape > 0) (Left "Expected nonempty MLX AdamW moments")
        pure (Finite, actual)

configuration :: AdamW -> Native.Value -> Either String ()
configuration expected observed = do
    fields <- exact ("MLX AdamW configuration", ["betas", "epsilon", "weight_decay", "bias_correction"]) observed
    betas <- field "betas" fields >>= Native.list >>= traverse Native.number
    actualEpsilon <- field "epsilon" fields >>= Native.number
    weightDecay <- field "weight_decay" fields >>= Native.number
    correction <- field "bias_correction" fields
    unless (betas == map toRational (coefficients expected) && actualEpsilon == toRational (epsilon expected) && weightDecay == toRational (decay expected) && correction == Native.Boolean True) (Left "MLX AdamW settings differ from the consumed input")

tensor :: (Text, [Integer], Integer) -> Native.Value -> Either String Native.Tensor
tensor (dtype, shape, width) (Native.Tensor observed) = do
    unless (Native.tensorType observed == "mlx.core.array" && Native.dtype observed == dtype && Native.shape observed == shape && Native.size observed == width * product shape && Native.layout observed == "mlx.row-major") (Left "MLX checkpoint tensor representation mismatch")
    pure observed
tensor _ _ = Left "Expected a native MLX checkpoint tensor"

rng :: Native.Value -> Either String Native.Tensor
rng = tensor ("mlx.core.uint32", [keyWords], Header.fp32Bytes)

keyWords, uint64Bytes :: Integer
keyWords = 2
uint64Bytes = 8
