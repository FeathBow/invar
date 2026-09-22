{-# LANGUAGE OverloadedStrings #-}

module Invar.Score.Output (Input (..), Body (..), LogRatio (..), observe, ratioValue) where

import Control.Monad (unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (FromJSON, Object, Value (..), object, withObject, (.:), (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.Foldable (traverse_)
import Data.Map.Strict qualified as Map
import Data.Ratio (denominator, numerator)
import Data.Text.Encoding (decodeUtf8)
import Data.Word (Word32)
import GHC.Float (castWord32ToFloat)
import Invar.Artifact qualified as Artifact
import Invar.Infer qualified as Infer
import Invar.Infer.Observation qualified as Inference
import Invar.Infer.Result qualified as Result
import Invar.Json qualified as Json
import Invar.Policy qualified as Policy
import Invar.Score.Execution qualified as Execution
import Invar.Score.Probe qualified as Probe
import Invar.Score.Resources qualified as Resources
import Invar.Spec.Score (LogRatio (..))
import Invar.Spec.Score qualified as S
import Numeric.Natural (Natural)

data Input = Input {source :: Inference.Report, target :: Infer.Request, inspection :: ByteString, materialization :: Policy.Description, probeSteps :: [Natural], measurements :: [Object]}

data Body = Body {probabilityWords :: [Word32], logRatio :: LogRatio, fullVocabulary :: Maybe S.Distribution, observation :: Value}
    deriving (Eq, Show)

observe :: Input -> Value -> Either String Body
observe inputs value = parseEither (withObject "cached path score" (parse inputs value)) value

parse :: Input -> Value -> Object -> Parser Body
parse inputs value fields = do
    let full = not (null (probeSteps inputs))
        additional = if full then ["source_inspection", "full_vocabulary", "measurements"] else []
    Json.fields (["format", "role", "use_admission", "source_inspection_sha256", "source", "target", "request", "prefix_tokens", "response_tokens", "log_probability_bits", "probability", "execution", "implementation"] ++ additional) fields
    equals fields "format" (if full then "invar-cached-distribution-probe-v1" else "invar-cached-path-score-v1" :: String)
    equals fields "role" (if full then "cached_behavior_full_vocabulary" else "cached_behavior_cross_score" :: String)
    equals fields "use_admission" ("not_evaluated" :: String)
    equals fields "source_inspection_sha256" (Artifact.hex (SHA256.hash (inspection inputs)))
    equals fields "source" (Inference.describe (source inputs))
    let original = Inference.result (source inputs)
        requested = target inputs
        (prefix, response) = splitAt (fromIntegral (Result.promptLength original)) (Result.tokens original)
    equals fields "prefix_tokens" prefix
    equals fields "response_tokens" response
    equals fields "request" (object ["prompt" .= Infer.prompt requested, "tokens" .= Infer.tokens requested, "temperature" .= Infer.temperature requested, "seed" .= Infer.seed requested])
    fields .: "target" >>= withObject "score materialization" (targetFields inputs)
    equals fields "probability" (object ["role" .= ("behavior" :: String), "log_base" .= ("e" :: String), "representation" .= ("F32 words" :: String), "zero_support_word" .= (0xff800000 :: Word32), "temperature" .= Infer.temperature requested, "mask" .= ("none" :: String), "top_k" .= ("disabled" :: String), "top_p" .= ("disabled" :: String)])
    relation <- fields .: "execution" >>= withObject "score execution" (Execution.observe (Execution.Expected (Result.truncated original) (length response) full))
    fields .: "implementation" >>= withObject "score implementation" implementation
    encoded <- fields .: "log_probability_bits"
    unless (length encoded == length response) (fail "Scored probabilities do not cover the complete prescribed response")
    values <- traverse probability encoded
    let ratio = case sequence values of
            Nothing -> PositiveInfinity
            Just finite -> Finite (sum (zipWith (\p q -> toRational p - q) (Result.behavior original) finite))
    vectors <-
        if full
            then do
                equals fields "source_inspection" (decodeUtf8 (inspection inputs))
                equals fields "measurements" (map Object (measurements inputs))
                Resources.observe relation (measurements inputs)
                Just <$> (fields .: "full_vocabulary" >>= Probe.observe relation (probeSteps inputs, zip response encoded))
            else pure Nothing
    pure (Body encoded ratio vectors value)

equals :: (Eq a, FromJSON a) => Object -> Key -> a -> Parser ()
equals fields key expected = do
    actual <- fields .: key
    unless (actual == expected) (fail ("Score differs at " ++ show key))

targetFields :: Input -> Object -> Parser ()
targetFields inputs fields = do
    Json.fields ["adapter", "tokenizer", "base", "assembly", "model", "revision", "numerics"] fields
    let requested = target inputs
        description = materialization inputs
    mapM_ (uncurry (equals fields)) [("adapter", Infer.artifact requested), ("tokenizer", Infer.tokenizer requested), ("base", Infer.base requested), ("assembly", Infer.assembly requested), ("model", Policy.model description), ("revision", Policy.revision description)]
    profile <- fields .: "numerics"
    when (null (profile :: String)) (fail "Missing numerical profile identity")

implementation :: Object -> Parser ()
implementation fields = do
    Json.fields ["sources_sha256", "packages"] fields
    sources <- fields .: "sources_sha256"
    packages <- fields .: "packages"
    when (Map.null (sources :: Map.Map String Value) || any null (Map.keys sources)) (fail "Missing implementation source identities")
    traverse_ Json.identity sources
    when (Map.null (packages :: Map.Map String String) || any null (Map.keys packages) || any null (Map.elems packages)) (fail "Missing implementation package versions")

probability :: Word32 -> Parser (Maybe Rational)
probability 0xff800000 = pure Nothing
probability encoded = do
    let value = castWord32ToFloat encoded
    when (isNaN value || isInfinite value || value > 0) (fail "Expected a nonpositive FP32 log probability or negative-infinite zero support")
    pure (Just (toRational value))

ratioValue :: LogRatio -> Value
ratioValue (Finite value) = object ["kind" .= ("finite" :: String), "numerator" .= numerator value, "denominator" .= denominator value]
ratioValue PositiveInfinity = object ["kind" .= ("positive_infinity" :: String)]
