{-# LANGUAGE OverloadedStrings #-}

module Invar.Numerical (
    Run (..),
    BoundRun (BoundRun, ScoredRun, ProbedRun, reference, candidate),
    Probe (..),
    Side (..),
    ObservationError (..),
    Scope,
    ScopeId,
    Observed,
    Path (..),
    Claim (..),
    Relation (..),
    Direction (..),
    Problem (..),
    Finding,
    observe,
    observeWith,
    admit,
    consistent,
    establish,
    finding,
    scope,
    scopeId,
    path,
    describe,
    describeFinding,
) where

import Control.Monad (unless)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value, encode, object, (.=))
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Map.Strict qualified as Map
import Data.Ratio (denominator, numerator)
import Invar.Artifact qualified as Artifact
import Invar.Evidence.Encoding qualified as EvidenceEncoding
import Invar.Infer qualified as Infer
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Observation qualified as Inference
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Result qualified as Result
import Invar.Infer.Session qualified as Session
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Numerical.Distribution qualified as KL
import Invar.Policy qualified as Policy
import Invar.Score qualified as Score
import Invar.Score.Output qualified as ScoreOutput
import Invar.Spec.Evidence qualified as Evidence
import Invar.Spec.Invocation qualified as Invocation
import Invar.Spec.Numerical (Claim (..), Direction (..), Observed, Path (..), Problem (..), Relation (..), Scope, ScopeId, Side (..), path, scope, scopeId)
import Invar.Spec.Numerical qualified as N
import Invar.Spec.Score qualified as S
import Numeric.Natural (Natural)
import System.Exit (ExitCode (..))

data Run = Run
    { planned :: Infer.Plan
    , binding :: Invocation.Binding
    , exitCode :: Int
    , logBytes :: ByteString
    , batch :: Maybe [Call.Call]
    }

data BoundRun
    = BoundRun {reference :: Run, candidate :: Run}
    | ScoredRun {reference :: Run, candidate :: Run, scoring :: [(Side, Score.Report)]}
    | ProbedRun {reference :: Run, candidate :: Run, scoring :: [(Side, Score.Report)], probing :: [Probe]}

data Probe = Probe {pathSource :: Side, implementation :: Side, probeReport :: Score.Report}

data ObservationError
    = ProcessFailed Side Int
    | InvalidRun Side String
    | IncomparableInputs String
    | InvalidScore Side String
    | InvalidProbe Side String
    deriving (Eq, Show)

data Finding = Finding Claim Evidence.Verdict
    deriving (Eq, Show)

establish :: Claim -> Observed -> Finding
establish target observed = Finding target (Evidence.check graph root)
  where
    root = Evidence.EvidenceId 0
    graph = Map.singleton root (Evidence.Node (Evidence.Numerical target) (Evidence.Observe observed))

finding :: Finding -> Evidence.Verdict
finding (Finding _ result) = result

observe :: BoundRun -> Either ObservationError Observed
observe supplied = do
    left <- admit Reference (reference supplied)
    right <- admit Candidate (candidate supplied)
    observeWith (left, right) supplied

observeWith :: (Inference.Report, Inference.Report) -> BoundRun -> Either ObservationError Observed
observeWith (left, right) supplied = do
    let before = Inference.result left
        after = Inference.result right
        requested = Result.consumed before
        other = Result.consumed after
        (prefix, generated) = splitTokens before
        (otherPrefix, otherGenerated) = splitTokens after
        same name equal = unless equal (Left (IncomparableInputs name))
    same "tokenizer" (Infer.tokenizer requested == Infer.tokenizer other)
    same "prompt" (Infer.prompt requested == Infer.prompt other)
    same "token budget" (Infer.tokens requested == Infer.tokens other)
    same "temperature" (Infer.temperature requested == Infer.temperature other)
    same "logical seed" (Infer.seed requested == Infer.seed other)
    same "tokenized prefix" (prefix == otherPrefix)
    leftSource <- source Reference left
    rightSource <- source Candidate right
    measuredScores <- checkedScores leftSource rightSource supplied
    probes <- checkedProbes leftSource rightSource supplied
    distributions <- concat <$> traverse (compareProbes probes) [Reference, Candidate]
    let measuredProbes = [(side, targetSide, Score.fact report) | (side, targetSide, report) <- probes]
        encoded = Lazy.toStrict (encode (scopeValue (leftSource, rightSource) prefix (measuredScores, measuredProbes)))
        selected = N.Scope (N.ScopeId (SHA256.hash encoded)) leftSource rightSource prefix measuredScores measuredProbes
        common = takeWhile (uncurry (==)) (zip generated otherGenerated)
        count = length common
        sameTokens = generated == otherGenerated
        ratio = sum (zipWith (\p q -> toRational p - toRational q) (take count (Result.behavior before)) (take count (Result.behavior after)))
        measured =
            Path
                { firstDivergence = if sameTokens then Nothing else Just (fromIntegral count)
                , matchingSteps = fromIntegral count
                , prefixLogRatio = ratio
                , referenceLength = fromIntegral (length generated)
                , candidateLength = fromIntegral (length otherGenerated)
                , referenceTruncated = Result.truncated before
                , candidateTruncated = Result.truncated after
                , tokensEqual = sameTokens
                , behaviorBitsEqual = sameTokens && Result.behaviorBits before == Result.behaviorBits after
                }
    pure (N.Observed selected measured distributions)

checkedScores :: N.Source -> N.Source -> BoundRun -> Either ObservationError [(Side, S.Fact)]
checkedScores left right supplied = concat <$> traverse select [Reference, Candidate]
  where
    attachments = case supplied of
        BoundRun {} -> []
        ScoredRun {scoring = values} -> values
        ProbedRun {scoring = values} -> values
    select side = case [Score.fact report | (selected, report) <- attachments, side == selected] of
        [] -> pure []
        [measured] -> do
            let (original, other) = case side of Reference -> (left, right); Candidate -> (right, left)
                target = S.target measured
                same name equal = unless equal (Left (InvalidScore side name))
            same "source execution" (S.source measured == original)
            same "target request" (S.sourceRequest target == N.sourceRequest other)
            same "target materialization" (S.sourcePolicy target == N.sourcePolicy other)
            pure [(side, measured)]
        _ -> Left (InvalidScore side "duplicate path scores")

checkedProbes :: N.Source -> N.Source -> BoundRun -> Either ObservationError [(Side, Side, Score.Report)]
checkedProbes left right supplied = concat <$> traverse select [(side, target) | side <- [Reference, Candidate], target <- [Reference, Candidate]]
  where
    attachments = case supplied of ProbedRun {probing = values} -> values; _ -> []
    sourceFor side = case side of Reference -> left; Candidate -> right
    select (side, targetSide) = case [value | Probe selected target value <- attachments, selected == side, target == targetSide] of
        [] -> pure []
        [report] -> do
            let expected = sourceFor targetSide
                measured = Score.fact report
                target = S.target measured
                same name equal = unless equal (Left (InvalidProbe side name))
            same "source execution" (S.source measured == sourceFor side)
            same "target request" (S.sourceRequest target == N.sourceRequest expected)
            same "target materialization" (S.sourcePolicy target == N.sourcePolicy expected)
            case Score.fullVocabulary report of
                Nothing -> Left (InvalidProbe side "missing full-vocabulary snapshots")
                Just _ -> pure [(side, targetSide, report)]
        _ -> Left (InvalidProbe side "duplicate probe target")

compareProbes :: [(Side, Side, Score.Report)] -> Side -> Either ObservationError [(Side, [N.Distribution])]
compareProbes measured side = case (select Reference, select Candidate) of
    (Just left, Just right) -> do
        unless (S.vocabulary left == S.vocabulary right) (Left (InvalidProbe side "different vocabulary widths"))
        unless (map S.step (S.snapshots left) == map S.step (S.snapshots right)) (Left (InvalidProbe side "different probe steps"))
        values <- traverse compareStep (zip (S.snapshots left) (S.snapshots right))
        pure [(side, values)]
    _ -> pure []
  where
    select target = lookup (side, target) [((pathSide, targetSide), value) | (pathSide, targetSide, value) <- measured] >>= Score.fullVocabulary
    compareStep (left, right) = do
        (forward, backward) <- first (InvalidProbe side) (KL.enclose (S.massWords left) (S.massWords right))
        pure (N.Distribution (S.step left) forward backward)

consistent :: [(FilePath, (Maybe FilePath, Int))] -> Either String ()
consistent supplied = mapM_ agreed (Map.elems (Map.fromListWith (++) [(located, [declared]) | (located, declared) <- supplied]))
  where
    agreed (selected : rest) = unless (all (== selected) rest) (Left "Runs that name the same log declare different batches or outcomes")
    agreed [] = pure ()

admit :: Side -> Run -> Either ObservationError Inference.Report
admit side run = do
    own <- first (InvalidRun side . show) (Call.prepare (binding run) (planned run))
    (protocol, declared) <- case batch run of
        Nothing -> pure (Session.Single, [own])
        Just calls -> do
            unless (any (\member -> (Call.binding member, Call.plan member) == (binding run, planned run)) calls) (Left (InvalidRun side "The run's declaration is not a member of its declared batch"))
            pure (Session.Batched, calls)
    admitted <- first failed (Replay.standalone protocol (Session.Declaration declared Nothing) (Replay.declared (exitCode run)) (logBytes run))
    case filter ((== binding run) . Trajectory.binding . Replay.trajectory) admitted of
        [single] -> pure (Inference.view single)
        _ -> Left (InvalidRun side "Expected one admitted inference for the run's binding")
  where
    failed (Session.Exited (ExitFailure code)) = ProcessFailed side code
    failed problem = InvalidRun side (show problem)

source :: Side -> Inference.Report -> Either ObservationError N.Source
source side report = do
    description <- first (InvalidRun side) (Inference.policyDescription report)
    pure (N.Source (Result.consumed (Inference.result report)) (Inference.binding report) description (Inference.logDigest report))

splitTokens :: Result.Result -> ([Natural], [Natural])
splitTokens result = splitAt (fromIntegral (Result.promptLength result)) (Result.tokens result)

scopeValue :: (N.Source, N.Source) -> [Natural] -> ([(Side, S.Fact)], [(Side, Side, S.Fact)]) -> Value
scopeValue (left, right) prefix (measured, probes) =
    object
        [ "format" .= ("invar-finite-path-scope-v1" :: String)
        , "reference" .= sourceValue left
        , "candidate" .= sourceValue right
        , "prefix_tokens" .= prefix
        , "scored_paths" .= map scoreValue measured
        , "full_vocabulary_probes" .= map probeValue probes
        ]

probeValue :: (Side, Side, S.Fact) -> Value
probeValue (side, targetSide, measured) = object ["path_source" .= show side, "implementation" .= show targetSide, "score" .= scoreValue (side, measured)]

scoreValue :: (Side, S.Fact) -> Value
scoreValue (side, measured) =
    object
        [ "path_source" .= show side
        , "source" .= sourceValue (S.source measured)
        , "target" .= sourceValue (S.target measured)
        , "source_minus_target_log_ratio" .= ScoreOutput.ratioValue (S.logRatio measured)
        , "target_probability_bits" .= S.probabilityWords measured
        , "checked_observation_json" .= Bytes.unpack (S.observation measured)
        ]

sourceValue :: N.Source -> Value
sourceValue selected =
    object
        [ "log_sha256" .= N.sourceLog selected
        , "model" .= Policy.model description
        , "revision" .= Policy.revision description
        , "adapter" .= Infer.artifact requested
        , "tokenizer" .= Infer.tokenizer requested
        , "base" .= Infer.base requested
        , "assembly" .= Infer.assembly requested
        , "prompt" .= Infer.prompt requested
        , "token_limit" .= Infer.tokens requested
        , "temperature" .= Infer.temperature requested
        , "seed" .= Infer.seed requested
        , "binding" .= object ["call" .= call, "attempt" .= attempt, "instance" .= instanceId]
        ]
  where
    requested = N.sourceRequest selected
    description = N.sourcePolicy selected
    Invocation.Binding (Invocation.CallId call) (Invocation.AttemptId attempt) (Invocation.Instance instanceId) = N.sourceBinding selected

describe :: Observed -> Value
describe observed =
    object
        [ "format" .= ("invar-finite-path-observation-v1" :: String)
        , "scope_sha256" .= Artifact.hex identity
        , "scope" .= scopeValue (left, right) prefix (measuredScores, measuredProbes)
        , "first_divergence_zero_based" .= firstDivergence measured
        , "matching_steps" .= matchingSteps measured
        , "common_prefix_log_ratio" .= object ["numerator" .= numerator ratio, "denominator" .= denominator ratio]
        , "reference_length" .= referenceLength measured
        , "candidate_length" .= candidateLength measured
        , "reference_truncated" .= referenceTruncated measured
        , "candidate_truncated" .= candidateTruncated measured
        , "tokens_equal" .= tokensEqual measured
        , "behavior_bits_equal" .= behaviorBitsEqual measured
        , "scored_paths" .= map scoreValue measuredScores
        , "full_vocabulary" .= map distributionValue (N.distributions observed)
        , "probability_role" .= ("reported selected-token behavior log-probabilities" :: String)
        , "log_ratio_scope" .= ("reference minus candidate; matching prefix and token only; exact rational sum of logged FP32 measurements, not exact model probabilities" :: String)
        , "unestablished" .= (["sampler coupling", "per-step cache lineage", "ideal model or actual sampler distribution KL", "population guarantee", "use admission"] :: [String])
        ]
  where
    N.Scope (N.ScopeId identity) left right prefix measuredScores measuredProbes = scope observed
    measured = path observed
    ratio = prefixLogRatio measured

distributionValue :: (Side, [N.Distribution]) -> Value
distributionValue (side, values) =
    object
        [ "path_source" .= show side
        , "probability_role" .= ("exact normalization of recorded FP32 behavior masses" :: String)
        , "reduction" .= ("outward integer fixed-point logarithm intervals; 96 fractional bits; 32 series terms" :: String)
        , "log_base" .= ("e" :: String)
        , "aggregation" .= ("every selected step must meet the budget" :: String)
        , "steps" .= [object ["step" .= N.step value, "kl_reference_candidate" .= bounds (N.forward value), "kl_candidate_reference" .= bounds (N.backward value)] | value <- values]
        ]
  where
    bounds KL.InfiniteKL = object ["kind" .= ("positive_infinity" :: String)]
    bounds (KL.FiniteBounds lower upper) = object ["kind" .= ("finite_interval" :: String), "lower" .= rational lower, "upper" .= rational upper]
    rational value = object ["numerator" .= numerator value, "denominator" .= denominator value]

describeFinding :: Finding -> Value
describeFinding (Finding (N.Claim selected relation) result) =
    object
        [ "scope_sha256" .= Artifact.hex identity
        , "relation" .= show relation
        , "strength" .= ("finite_observation" :: String)
        , "use_admission" .= ("not_evaluated" :: String)
        , "judgement" .= EvidenceEncoding.verdict witnessValue result
        ]
  where
    N.ScopeId identity = scopeId selected
    witnessValue (Evidence.NumericalWitness observed) = describe observed
    witnessValue (Evidence.OutputWitness _ _) = object ["kind" .= ("output_bytes" :: String)]
    witnessValue (Evidence.TaskLossWitness _) = object ["kind" .= ("task_loss" :: String)]
