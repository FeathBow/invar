{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Probability (Sample (..), validate, observe) where

import Control.Monad (unless, when)
import Data.Aeson (Object, Value, withObject, (.:))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.List (sortOn)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Word (Word32)
import Invar.Float32 qualified as Float32
import Invar.Infer.Wire qualified as Binding
import Invar.Json qualified as Json
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
    observed <- observe (intended, P.checkedRequest expected) encoded
    unless (Map.fromList [(sampleName entry, proximal entry) | entry <- observed] == S.proximals streamed) (Left "Probability artifact proximal words differ from the reported steps")
    when (S.sourceOf streamed == S.FromLearner) $
        unless (Map.fromList [(sampleName entry, reported) | entry <- observed, Just reported <- [reference entry]] == S.referencesOf streamed) (Left "Probability artifact reference words differ from the reported steps")
    unless (sortOn (\(position, name, _) -> (position, name)) [(position, sampleName entry, words32) | entry <- observed, (position, words32) <- currents entry] == sortOn (\(position, name, _) -> (position, name)) (S.currents streamed)) (Left "Probability artifact current words differ from the reported steps")

data Sample = Sample {sampleObject :: Object, sampleName :: Text, behavior :: [Word32], proximal :: [Word32], reference :: Maybe [Word32], engineReference :: [Word32], currents :: [(Natural, [Word32])]}

observe :: (Value, Request.Request) -> ByteString -> Either String [Sample]
observe (intended, consumed) encoded = Json.decode encoded >>= parseEither (withObject "probability observation" (document (intended, consumed)))

document :: (Value, Request.Request) -> Object -> Parser [Sample]
document (intended, consumed) fields = do
    exact ["format", "invocation", "request", "samples"] fields
    format <- fields .: "format"
    unless (format == ("invar-probabilities-v5" :: Text)) (fail "Unknown probability observation format")
    invocation <- fields .: "invocation"
    unless (invocation == intended) (fail "Probability observation invocation mismatch")
    request <- fields .: "request"
    unless (request == Request.value consumed) (fail "Probability observation request mismatch")
    observations <- fields .: "samples"
    withObject "probability request" (samples observations) (Request.value consumed)

samples :: [Object] -> Object -> Parser [Sample]
samples observed request = do
    order <- request .: "order" :: Parser [Text]
    plan <- request .: "steps" :: Parser [[Text]]
    source <- request .: "reference_source" :: Parser Text
    delivered <- request .: "samples" :: Parser [Object]
    originals <- Map.fromList <$> traverse (\item -> (,) <$> item .: "sample" <*> ((,) <$> item .: "behavior_bits" <*> item .: "reference_bits")) delivered
    names <- traverse (.: "sample") observed
    unless (names == order) (fail "Probability samples differ from logical order")
    let learnerSource = source == ("learner" :: Text)
        participation name = [position | (position, batch) <- zip [0 ..] plan, name `elem` batch]
        first = case plan of
            batch : _ -> batch
            [] -> []
    traverse (\(name, item) -> maybe (fail "Unknown probability sample") (\(consumedWords, engineWords) -> sample learnerSource (participation name, name `elem` first) consumedWords engineWords item) (Map.lookup name originals)) (zip names observed)

sample :: Bool -> ([Natural], Bool) -> [Word32] -> [Word32] -> Object -> Parser Sample
sample learnerSource (participation, inFirst) behaviorWords engineWords fields = do
    exact (["sample", "dtype", "proximal", "steps"] ++ ["reference" | learnerSource]) fields
    name <- fields .: "sample"
    dtype <- fields .: "dtype"
    unless (dtype == ("F32" :: Text)) (fail "Update probability observations must use FP32")
    proximalWords <- vector (length behaviorWords) fields "proximal"
    referenceWords <- if learnerSource then Just <$> vector (length behaviorWords) fields "reference" else pure Nothing
    entries <- fields .: "steps" :: Parser [Object]
    observed <- traverse entry entries
    unless (map fst observed == participation) (fail "Probability steps differ from the declared mini-batches of this sample")
    case observed of
        (0, currentWords) : _ | inFirst -> unless (currentWords == proximalWords) (fail "Proximal words differ from the first step's observation")
        _ -> pure ()
    pure (Sample fields name behaviorWords proximalWords referenceWords engineWords observed)
  where
    entry item = do
        exact ["step", "current"] item
        (,) <$> item .: "step" <*> vector (length behaviorWords) item "current"

vector :: Int -> Object -> Key -> Parser [Word32]
vector count fields role = do
    encoded <- fields .: role :: Parser [Word32]
    unless (length encoded == count && count > 0) (fail "Probability vector token count mismatch")
    unless (all Float32.logProbability encoded) (fail "Invalid probability floating-point words")
    pure encoded

exact :: [Key] -> Object -> Parser ()
exact names fields = unless (Set.fromList (Fields.keys fields) == Set.fromList names) (fail "Unexpected probability observation fields")
