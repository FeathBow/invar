{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE Trustworthy #-}

module Invar.Paths (
    TokenPath (..),
    PathId (..),
    pathId,
    Rule (..),
    ForwardWeights (..),
    ForwardId (..),
    forwardId,
    ImplementationId (..),
    Role (..),
    Provenance (..),
    Input (..),
    Side (..),
    Observation (..),
    Relation (..),
    Claim (..),
    Difference (..),
    Gap (..),
    Problem (..),
    Judgement (..),
    ScopeId (..),
    Observed (..),
    ScopedClaim (..),
    scoped,
    missingScope,
    judgeScoped,
    policyDifference,
    describeObserved,
    describeObservation,
    UpdateObservations (..),
    fromUpdate,
    fromSamples,
    provenanceName,
    measure,
    measureAll,
    judge,
    describeGap,
) where

import Prelude hiding (words)

import Control.Monad (unless)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value (..), encode, object, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Lazy qualified as Lazy
import Data.Map.Strict qualified as Map
import Data.Ratio (denominator, numerator)
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.Encoding qualified as Text
import Data.Word (Word32)
import Invar.Artifact qualified as Artifact
import Invar.Canonical qualified as Canonical
import Invar.Digest qualified as Digest
import Invar.Float32 qualified as Float32
import Invar.Learn.Mismatch qualified as Mismatch
import Invar.Learn.Probability qualified as Probability
import Invar.Learn.Report qualified as Report
import Invar.Learn.Request qualified as Request
import Invar.Policy qualified as Policy
import Numeric.Natural (Natural)

data TokenPath = TokenPath {prefix :: [Natural], response :: [Natural]}
    deriving (Eq, Show)

newtype PathId = PathId ByteString
    deriving (Eq, Ord, Show)

pathId :: TokenPath -> PathId
pathId (TokenPath prefix response) = PathId (SHA256.hash (Canonical.encode (Canonical.Object (Map.fromList [("format", Canonical.Text "invar-token-path-v1"), ("prefix_length", natural (length prefix)), ("tokens", Canonical.Array (map natural (prefix ++ response)))]))))

data Rule = Rule {temperature :: Double, logprobs :: Text, truncation :: Text}
    deriving (Eq, Show)

data ForwardWeights = ForwardWeights {base :: String, adapter :: String}
    deriving (Eq, Show)

newtype ForwardId = ForwardId ByteString
    deriving (Eq, Ord, Show)

forwardId :: ForwardWeights -> ForwardId
forwardId (ForwardWeights base adapter) = ForwardId (SHA256.hash (Canonical.encode (Canonical.Object (Map.fromList [("format", Canonical.Text "invar-forward-weights-v1"), ("base", text base), ("adapter", text adapter)]))))

newtype ImplementationId = ImplementationId ByteString
    deriving (Eq, Show)

data Role = Proximal | Current | Reference
    deriving (Eq, Show)

data Provenance = Rollout Natural Text | Learner Natural Natural Text Role
    deriving (Eq, Show)

data UpdateObservations = UpdateObservations
    { proximal :: [(Observation, Observation)]
    , references :: [(Observation, Observation)]
    , currents :: [(Observation, Observation)]
    }
    deriving (Eq, Show)

data Input = Input
    { inputPrompt :: Text
    , inputSeed :: Integer
    , inputLimit :: Natural
    , inputTemperature :: Double
    }
    deriving (Eq, Show)

data Side = Side
    { sideModel :: String
    , sideRevision :: String
    , sideTokenizer :: String
    , sideBase :: String
    , sideAssembly :: String
    }
    deriving (Eq, Show)

data Observation = Observation
    { path :: PathId
    , rule :: Rule
    , input :: Input
    , forward :: ForwardWeights
    , implementation :: ImplementationId
    , implementationPolicy :: Maybe Policy.Description
    , words :: [Word32]
    , provenance :: Provenance
    }
    deriving (Eq, Show)

data Relation = SameWords | PerTokenLogGapWithin Rational | PathLogRatioWithin Rational
    deriving (Eq, Show)

newtype Claim = Claim {relation :: Relation}
    deriving (Eq, Show)

data Difference = Finite Rational | PositiveInfinity
    deriving (Eq, Show)

data Gap = Gap
    { tokens :: Int
    , identical :: Int
    , summary :: Maybe Mismatch.Summary
    , largest :: Difference
    , ratio :: Difference
    }
    deriving (Eq, Show)

data Problem
    = ScopeMismatch ScopeId ScopeId
    | PathDiffer PathId PathId
    | RuleDiffer Rule Rule
    | ForwardDiffer ForwardWeights ForwardWeights
    | LengthDiffer Natural Natural
    | EmptyPath
    | InvalidWord Natural Word32
    | InvalidBudget Rational
    deriving (Eq, Show)

data Judgement = Satisfied Gap | Violated Gap | Insufficient Problem
    deriving (Eq, Show)

newtype ScopeId = ScopeId ByteString
    deriving (Eq, Ord, Show)

data Observed = Observed {scope :: ScopeId, learner :: Observation, engine :: Observation}
    deriving (Eq, Show)

data ScopedClaim = ScopedClaim {scopeOf :: ScopeId, relationOf :: Relation}
    deriving (Eq, Show)

scoped :: Observation -> Observation -> Observed
scoped learner engine = Observed identity learner engine
  where
    identity = ScopeId (SHA256.hash (Lazy.toStrict (encode (object ["format" .= ("invar-path-observation-v1" :: Text), "learner" .= describeObservation learner, "engine" .= describeObservation engine]))))

missingScope :: (Show key) => key -> Relation -> ScopeId
missingScope key relation = ScopeId (SHA256.hash (Text.encodeUtf8 (Text.pack (show (key, relation)))))

judgeScoped :: ScopedClaim -> Observed -> Judgement
judgeScoped claim observed
    | scopeOf claim /= scope observed = Insufficient (ScopeMismatch (scopeOf claim) (scope observed))
    | otherwise = judge (Claim (relationOf claim)) (learner observed) (engine observed)

describeObserved :: Observed -> Value
describeObserved observed = object ["format" .= ("invar-path-observation-v1" :: Text), "scope_sha256" .= scopeText (scope observed), "input" .= describeInput (input (learner observed)), "learner" .= describeObservation (learner observed), "engine" .= describeObservation (engine observed)]

describeObservation :: Observation -> Value
describeObservation observation =
    object
        [ "path_sha256" .= pathText (path observation)
        , "rule" .= object ["temperature" .= temperature (rule observation), "logprobs" .= logprobs (rule observation), "truncation" .= truncation (rule observation)]
        , "input" .= describeInput (input observation)
        , "forward_weights" .= object ["base" .= base (forward observation), "adapter" .= adapter (forward observation), "identity_sha256" .= forwardText (forwardId (forward observation))]
        , "implementation_sha256" .= implementationText (implementation observation)
        , "policy" .= fmap policyValue (implementationPolicy observation)
        , "words" .= words observation
        , "provenance" .= provenanceValue (provenance observation)
        ]

describeInput :: Input -> Value
describeInput selected = object ["prompt" .= inputPrompt selected, "seed" .= inputSeed selected, "limit" .= inputLimit selected, "temperature" .= inputTemperature selected]

policyValue :: Policy.Description -> Value
policyValue policy = object ["model" .= Policy.model policy, "revision" .= Policy.revision policy, "adapter" .= Policy.adapter policy, "tokenizer" .= Policy.tokenizer policy, "base" .= Policy.base policy, "assembly" .= Policy.assembly policy]

provenanceValue :: Provenance -> Value
provenanceValue (Rollout version name) = object ["kind" .= ("rollout" :: Text), "version" .= version, "sample" .= name]
provenanceValue (Learner update step name role) = object ["kind" .= ("learner" :: Text), "update" .= update, "step" .= step, "sample" .= name, "role" .= roleText role]

provenanceName :: Observation -> Text
provenanceName observation = case provenance observation of
    Rollout _ name -> name
    Learner _ _ name _ -> name

roleText :: Role -> Text
roleText Proximal = "proximal"
roleText Current = "current"
roleText Reference = "reference"

pathText :: PathId -> String
pathText (PathId identity) = Artifact.hex identity

forwardText :: ForwardId -> String
forwardText (ForwardId identity) = Artifact.hex identity

implementationText :: ImplementationId -> String
implementationText (ImplementationId identity) = Artifact.hex identity

scopeText :: ScopeId -> String
scopeText (ScopeId identity) = Artifact.hex identity

fromUpdate :: Side -> Side -> Report.Report -> ByteString -> [(Natural, Text, String)] -> Either String UpdateObservations
fromUpdate learner engine report encoded states = do
    samples <- Probability.observe (Report.invocation report, Report.checkedRequest report) encoded
    fromSamples learner engine report samples states

fromSamples :: Side -> Side -> Report.Report -> [Probability.Sample] -> [(Natural, Text, String)] -> Either String UpdateObservations
fromSamples learnerIdentity engineIdentity report probabilities currentStates = parseEither (withObject "update path observations" (document probabilities)) (Request.value (Report.checkedRequest report))
  where
    states = Map.fromList [((step, sample), state) | (step, sample, state) <- currentStates]
    document samples fields = do
        policy <- fields .: "policy" :: Parser String
        reference <- fields .: "reference" :: Parser String
        learnerBase <- fields .: "base" :: Parser String
        learnerAssembly <- fields .: "assembly" :: Parser String
        learnerTokenizer <- fields .: "tokenizer" :: Parser String
        (engineBase, engineAssembly) <- fields .: "behavior_model" >>= withObject "behavior model" (\model -> (,) <$> model .: "base" <*> model .: "assembly") :: Parser (String, String)
        unless (learnerBase == sideBase learnerIdentity && learnerAssembly == sideAssembly learnerIdentity && learnerTokenizer == sideTokenizer learnerIdentity) (fail "Learner path identity differs from the update request")
        unless (engineBase == sideBase engineIdentity && engineAssembly == sideAssembly engineIdentity) (fail "Engine path identity differs from the update request")
        update <- fields .: "schedule" >>= withObject "update schedule" (.: "update") :: Parser Natural
        delivered <- fields .: "samples" :: Parser [Value]
        observations <- traverse (sample update policy reference samples) delivered
        pure (UpdateObservations (concatMap proximalOf observations) (concatMap referencesOf observations) (concatMap currentsOf observations))
      where
        byName = Map.fromList [(Probability.sampleName probability, probability) | probability <- samples]
        sample update policy reference _ = withObject "update path sample" inspect
          where
            inspect sampleFields = do
                name <- sampleFields .: "sample" :: Parser Text
                version <- sampleFields .: "version" :: Parser Natural
                prompt <- sampleFields .: "prompt" :: Parser Text
                seed <- sampleFields .: "seed" :: Parser Integer
                limit <- sampleFields .: "limit" :: Parser Natural
                tokens <- sampleFields .: "tokens" :: Parser [Natural]
                prefix <- sampleFields .: "prompt_length" :: Parser Natural
                temperature <- sampleFields .: "temperature" :: Parser Double
                behaviorPolicy <- sampleFields .: "behavior_policy" :: Parser String
                let rule = Rule temperature "processed" "none"
                    selectedInput = Input prompt seed limit temperature
                enginePolicy <- either fail pure (engineDescription engineIdentifier behaviorPolicy)
                referenceEnginePolicy <- either fail pure (engineDescription engineIdentifier reference)
                learnerPolicy <- either fail pure (learnerDescription learnerIdentifier policy)
                referenceLearnerPolicy <- either fail pure (learnerDescription learnerIdentifier reference)
                probability <- maybe (fail "Probability sample missing from the admitted update") pure (Map.lookup name byName)
                let tokenPath = TokenPath (take (fromIntegral prefix) tokens) (drop (fromIntegral prefix) tokens)
                    learnerImplementation = implementationId "learner" (sideBase learnerIdentity) (sideAssembly learnerIdentity)
                    engineImplementation = implementationId "engine" (sideBase engineIdentity) (sideAssembly engineIdentity)
                    learner step role adapter description words32 = Observation (pathId tokenPath) rule selectedInput (ForwardWeights (sideBase learnerIdentity) adapter) learnerImplementation (Just description) words32 (Learner update step name role)
                    engine adapter description words32 = Observation (pathId tokenPath) rule selectedInput (ForwardWeights (sideBase engineIdentity) adapter) engineImplementation (Just description) words32 (Rollout version name)
                    proximalPair = (learner 0 Proximal policy learnerPolicy (Probability.proximal probability), engine behaviorPolicy enginePolicy (Probability.behavior probability))
                    referencePairs = case Probability.reference probability of
                        Nothing -> []
                        Just words32 -> [(learner 0 Reference reference referenceLearnerPolicy words32, engine reference referenceEnginePolicy (Probability.engineReference probability))]
                    currentPair (step, words32) = do
                        state <- maybe (fail "Current step has no applied state") pure (Map.lookup (step, name) states)
                        currentPolicy <- either fail pure (learnerDescription learnerIdentifier state)
                        pure (learner step Current state currentPolicy words32, engine behaviorPolicy enginePolicy (Probability.behavior probability))
                currentPairs <- traverse currentPair (Probability.currents probability)
                pure (proximalPair, referencePairs, currentPairs)
        learnerIdentifier = (sideModel learnerIdentity, sideRevision learnerIdentity)
        engineIdentifier = (sideModel engineIdentity, sideRevision engineIdentity)
        learnerDescription (model, revision) adapter = Policy.describe (model, revision) (adapter, sideTokenizer learnerIdentity, sideBase learnerIdentity, sideAssembly learnerIdentity)
        engineDescription (model, revision) adapter = Policy.describe (model, revision) (adapter, sideTokenizer engineIdentity, sideBase engineIdentity, sideAssembly engineIdentity)
        proximalOf (pair, _, _) = [pair]
        referencesOf (_, pairs, _) = pairs
        currentsOf (_, _, pairs) = pairs

measure :: Observation -> Observation -> Either Problem Gap
measure left right = measureAll [(left, right)]

measureAll :: [(Observation, Observation)] -> Either Problem Gap
measureAll observations = do
    pairs <- concat <$> traverse checkedPair observations
    summarizePairs pairs

checkedPair :: (Observation, Observation) -> Either Problem [(Word32, Word32)]
checkedPair (left, right)
    | path left /= path right = Left (PathDiffer (path left) (path right))
    | rule left /= rule right = Left (RuleDiffer (rule left) (rule right))
    | length (words left) /= length (words right) = Left (LengthDiffer (fromIntegral (length (words left))) (fromIntegral (length (words right))))
    | null (words left) = Left EmptyPath
    | otherwise = traverse checked (zip3 [0 ..] (words left) (words right))
  where
    checked (index, leftWord, rightWord)
        | admissible leftWord && admissible rightWord = Right (leftWord, rightWord)
        | not (admissible leftWord) = Left (InvalidWord index leftWord)
        | otherwise = Left (InvalidWord index rightWord)
    admissible value = Float32.logProbability value || value == zeroSupport

summarizePairs :: [(Word32, Word32)] -> Either Problem Gap
summarizePairs pairs
    | null pairs = Left EmptyPath
    | any infinite pairs = Right (Gap count identical Nothing PositiveInfinity PositiveInfinity)
    | otherwise = case Mismatch.summarize [(map snd pairs, map fst pairs)] of
        Left _ -> Left EmptyPath
        Right summary ->
            let ratio = Mismatch.mean summary * fromIntegral (Mismatch.tokens summary)
             in Right (Gap count identical (Just summary) (Finite (Mismatch.largest summary)) (Finite ratio))
  where
    count = length pairs
    identical = length (filter (uncurry (==)) pairs)
    infinite (leftWord, rightWord) = (leftWord == zeroSupport) /= (rightWord == zeroSupport)

describeGap :: Gap -> Value
describeGap gap = case summary gap of
    Just summarized -> augment (Mismatch.describe summarized)
    Nothing -> object ["tokens" .= tokens gap, "identical" .= identical gap, "gap" .= Null, "path_log_ratio" .= differenceValue (ratio gap)]
  where
    augment (Object fields) = Object (Fields.insert "path_log_ratio" (differenceValue (ratio gap)) fields)
    augment value = value

differenceValue :: Difference -> Value
differenceValue (Finite value) = object ["kind" .= ("finite" :: Text), "numerator" .= numerator value, "denominator" .= denominator value]
differenceValue PositiveInfinity = object ["kind" .= ("positive_infinity" :: Text)]

implementationId :: Text -> String -> String -> ImplementationId
implementationId role base assembly = ImplementationId (SHA256.hash (Canonical.encode (Canonical.Object (Map.fromList [("format", Canonical.Text "invar-path-implementation-v1"), ("role", Canonical.Text role), ("base", text base), ("assembly", text assembly)]))))

judge :: Claim -> Observation -> Observation -> Judgement
judge claim left right
    | path left /= path right = Insufficient (PathDiffer (path left) (path right))
    | rule left /= rule right = Insufficient (RuleDiffer (rule left) (rule right))
    | not (comparable left right) = Insufficient (ForwardDiffer (forward left) (forward right))
    | invalid (relation claim) = Insufficient (InvalidBudget (budget (relation claim)))
    | otherwise = case measure left right of
        Left problem -> Insufficient problem
        Right gap -> decide gap
  where
    comparable learner engine = valid (forward learner) && valid (forward engine) && forward learner == forward engine
    invalid (PerTokenLogGapWithin value) = value < 0
    invalid (PathLogRatioWithin value) = value < 0
    invalid SameWords = False
    budget (PerTokenLogGapWithin value) = value
    budget (PathLogRatioWithin value) = value
    budget SameWords = 0
    decide gap = case relation claim of
        SameWords -> verdict (identical gap == tokens gap) gap
        PerTokenLogGapWithin limit -> verdict (within limit (largest gap)) gap
        PathLogRatioWithin limit -> verdict (within limit (ratio gap)) gap
    verdict True gap = Satisfied gap
    verdict False gap = Violated gap

policyDifference :: Relation -> Observation -> Observation -> Bool
policyDifference relation left right =
    path left == path right
        && rule left == rule right
        && valid (forward left)
        && valid (forward right)
        && forward left /= forward right
        && not (invalid relation)
  where
    invalid (PerTokenLogGapWithin value) = value < 0
    invalid (PathLogRatioWithin value) = value < 0
    invalid SameWords = False

within :: Rational -> Difference -> Bool
within _ PositiveInfinity = False
within limit (Finite value) = abs value <= limit

valid :: ForwardWeights -> Bool
valid (ForwardWeights base adapter) = Digest.sha256 base && Digest.sha256 adapter

zeroSupport :: Word32
zeroSupport = 0xff800000

natural :: (Integral value) => value -> Canonical.Value
natural = Canonical.Integer . toInteger

text :: String -> Canonical.Value
text = Canonical.Text . Text.pack
