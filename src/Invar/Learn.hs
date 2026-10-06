{-# LANGUAGE RoleAnnotations #-}

module Invar.Learn (Optimizer (..), Schedule (..), ReferenceSource (..), Settings (..), Plan, Error (..), synchronous, validate, materialization, prepare, observedInput, input, program, emission, rollout, invocation) where

import Control.Monad (unless)
import Data.Aeson (encode)
import Data.ByteString (ByteString)
import Data.ByteString.Lazy qualified as Lazy
import Data.Char (ord)
import Data.List (sortOn)
import Data.Map.Strict qualified as Map
import Data.Text qualified as Text
import Invar.Cohort qualified as Cohort
import Invar.Construct qualified as C
import Invar.Digest qualified as Digest
import Invar.Infer qualified as I
import Invar.Infer.Output qualified as Output
import Invar.Infer.Result qualified as Result
import Invar.Infer.Trajectory (Trajectory)
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Learn.Program qualified as P
import Invar.Learn.Stream (ReferenceSource (..))
import Invar.Learn.Stream qualified as S
import Invar.Learn.Wire qualified as Wire
import Invar.Materialization qualified as Materialization
import Invar.Rollout qualified as R
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as Load
import Invar.Spec.Program (Source (Semantic))
import Invar.Spec.Value (Scalar (..), Value (..))
import Numeric.Natural (Natural)

data Optimizer = Optimizer {learningRate :: Double, firstMoment :: Double, secondMoment :: Double, epsilon :: Double, weightDecay :: Double}
    deriving (Eq, Show)

data Schedule = Schedule {update :: Natural, staleness :: Natural, version :: Natural, behaviorPolicy :: String}
    deriving (Eq, Show)

synchronous :: Natural -> String -> Schedule
synchronous index current = Schedule {update = index, staleness = 0, version = index, behaviorPolicy = current}

data Settings = Settings
    { policy :: String
    , learner :: String
    , reference :: String
    , tokenizer :: String
    , base :: String
    , assembly :: String
    , behaviorBase :: String
    , behaviorAssembly :: String
    , clip :: Double
    , penalty :: Double
    , delta :: Double
    , steps :: Natural
    , optimizer :: Optimizer
    , schedule :: Schedule
    , referenceSource :: ReferenceSource
    }
    deriving (Eq, Show)

data InputSample = InputSample {sampleGroup :: String, sampleTrajectory :: Trajectory, sampleReward :: Rational}

type role Plan nominal
data Plan scope = Plan {planBatch :: R.Batch scope, planChecked :: A.Checked, planWorld :: E.World, planEmission :: E.Emission, planInput :: ByteString}

data Error = InvalidSettings String | PolicyMismatch | ReferenceMismatch | TokenizerMismatch | MaterializationMismatch | Construction C.BuildError | Evaluation E.Error | Lowering Wire.Error | Lifecycle V.Error | InvalidEmission
    deriving (Eq, Show)

prepare :: Settings -> R.Batch scope -> Either Error (Plan scope)
prepare settings batch = do
    let samples = [InputSample (R.group sample) (R.trajectory sample) (R.reward sample) | sample <- R.samples batch]
    (checked, sources, command, payload) <- compileInput settings samples
    pure Plan {planBatch = batch, planChecked = checked, planWorld = sources, planEmission = command, planInput = payload}

observedInput :: Settings -> Cohort.Batch scope -> Either Error (ByteString, ByteString)
observedInput settings batch = do
    let samples = [InputSample (Cohort.group (Cohort.source sample)) (Cohort.observed sample) (Cohort.reward sample) | sample <- Cohort.observations batch]
    (checked, _, _, payload) <- compileInput settings samples
    pure (A.bytes checked, payload)

compileInput :: Settings -> [InputSample] -> Either Error (A.Checked, E.World, E.Emission, ByteString)
compileInput settings samples = do
    validate settings
    unless (all ((== behaviorPolicy (schedule settings)) . I.artifact . Trajectory.request . sampleTrajectory) samples) (Left PolicyMismatch)
    mapM_ (materialization settings . Trajectory.request . sampleTrajectory) samples
    unless (all (scoredBy settings . Trajectory.reference . sampleTrajectory) samples) (Left ReferenceMismatch)
    checked <- either (Left . Construction) Right P.checked
    let sources = world settings samples
    commands <- either (Left . Evaluation) Right (A.run checked sources)
    case commands of
        [command] -> do
            payload <- either (Left . Lowering) Right (Wire.lower command)
            pure (checked, sources, command, Lazy.toStrict (encode payload))
        _ -> Left InvalidEmission

scoredBy :: Settings -> Maybe Output.Scored -> Bool
scoredBy settings scored
    | reference settings == behaviorPolicy (schedule settings) = null scored
    | otherwise = fmap Output.adapter scored == Just (reference settings)

validate :: Settings -> Either Error ()
validate settings = do
    unless (all identity [policy settings, learner settings, reference settings, base settings, assembly settings, behaviorBase settings, behaviorAssembly settings]) (Left (InvalidSettings "Expected lowercase SHA-256 artifact identities"))
    unless (identity (tokenizer settings)) (Left (InvalidSettings "Expected a lowercase SHA-256 tokenizer identity"))
    unless algorithm (Left (InvalidSettings "Invalid GRPO coefficients"))
    unless (steps settings > 0) (Left (InvalidSettings "An update needs at least one optimizer step"))
    unless (identity (behaviorPolicy chosenSchedule)) (Left (InvalidSettings "Expected a lowercase SHA-256 behavior policy identity"))
    unless (version chosenSchedule == (if update chosenSchedule > staleness chosenSchedule then update chosenSchedule - staleness chosenSchedule else 0)) (Left (InvalidSettings "Samples must come from version max(0, update - staleness)"))
    unless (version chosenSchedule /= update chosenSchedule || behaviorPolicy chosenSchedule == policy settings) (Left (InvalidSettings "Samples of the update's own version must come from the policy being updated"))
    unless adamw (Left (InvalidSettings "Invalid AdamW coefficients"))
  where
    algorithm = finite (clip settings) && clip settings > 0 && clip settings < 1 && nonnegative (penalty settings) && positive (delta settings)
    adamw = nonnegative (learningRate chosen) && positive (epsilon chosen) && nonnegative (weightDecay chosen) && all moment [firstMoment chosen, secondMoment chosen]
    chosen = optimizer settings
    chosenSchedule = schedule settings
    identity = Digest.sha256
    finite value = not (isNaN value || isInfinite value)
    positive value = finite value && value > 0
    nonnegative value = finite value && value >= 0
    moment value = nonnegative value && value < 1

materialization :: Settings -> I.Request -> Either Error ()
materialization settings requested = do
    unless (tokenizer settings == I.tokenizer requested) (Left TokenizerMismatch)
    unless (behaviorBase settings == I.base requested && behaviorAssembly settings == I.assembly requested) (Left MaterializationMismatch)

world :: Settings -> [InputSample] -> E.World
world settings samples = Map.fromList [(Semantic "policy", Load.imageValue image), (Semantic "learner", learnerValue settings), (Semantic "reference", text (reference settings)), (Semantic "reference_source", text (Text.unpack (S.sourceName (referenceSource settings)))), (Semantic "algorithm", algorithm), (Semantic "trajectories", keyed (trajectory . Trajectory.result . sampleTrajectory)), (Semantic "behavior_model", behaviorModel), (Semantic "schedule", record [("update", Atom (Token (update chosenSchedule))), ("staleness", Atom (Token (staleness chosenSchedule)))]), (Semantic "generations", keyed (const generation)), (Semantic "behavior", keyed (Sequence . map (Atom . Bits32) . Trajectory.behaviorBits . sampleTrajectory)), (Semantic "reference_scores", keyed (Sequence . map (Atom . Bits32) . maybe [] Output.scores . Trajectory.reference . sampleTrajectory)), (Semantic "rewards", keyed (Atom . Number . sampleReward)), (Semantic "groups", groupValues indexed), (Semantic "order", Sequence [Mapping (Map.singleton index marker) | (index, _) <- indexed])]
  where
    image = Materialization.learning (policy settings, learner settings, tokenizer settings, base settings, assembly settings, reference settings)
    behaviorModel = record [("base", text (behaviorBase settings)), ("assembly", text (behaviorAssembly settings))]
    chosenSchedule = schedule settings
    generation = record [("version", Atom (Token (version chosenSchedule))), ("policy", text (behaviorPolicy chosenSchedule))]
    indexed = zip [0 ..] samples
    keyed project = Mapping (Map.fromList [(index, project sample) | (index, sample) <- indexed])
    algorithm = record [("epsilon", number (clip settings)), ("penalty", number (penalty settings)), ("delta", number (delta settings)), ("steps", Atom (Number (fromIntegral (steps settings))))]

learnerValue :: Settings -> Value Natural
learnerValue settings = record [("policy", text (policy settings)), ("learner", text (learner settings)), ("tokenizer", text (tokenizer settings)), ("base", text (base settings)), ("assembly", text (assembly settings)), ("optimizer", parameters)]
  where
    chosen = optimizer settings
    parameters = record [("learning_rate", number (learningRate chosen)), ("betas", Sequence (map number [firstMoment chosen, secondMoment chosen])), ("epsilon", number (epsilon chosen)), ("weight_decay", number (weightDecay chosen))]

trajectory :: Result.Result -> Value Natural
trajectory observed = record [("prompt", text (I.prompt requested)), ("seed", Atom (Number (fromInteger (I.seed requested)))), ("limit", Atom (Token (I.tokens requested))), ("temperature", number (I.temperature requested)), ("tokens", Sequence (map (Atom . Token) (Result.tokens observed))), ("prompt_length", Atom (Token (Result.promptLength observed))), ("text", text (Result.response observed)), ("truncated", Atom (Boolean (Result.truncated observed)))]
  where
    requested = Result.consumed observed

groupValues :: [(Natural, InputSample)] -> Value Natural
groupValues samples = Sequence [Mapping members | (_, members) <- sortOn fst (Map.elems grouped)]
  where
    grouped = foldl' insert Map.empty samples
    insert groups (index, sample) = Map.insert (sampleGroup sample) updated groups
      where
        updated = case Map.lookup (sampleGroup sample) groups of
            Nothing -> (Map.size groups, Map.singleton index marker)
            Just (position, members) -> (position, Map.insert index marker members)

marker :: Value Natural
marker = Atom (Boolean True)

record :: [(String, Value Natural)] -> Value Natural
record = Record . Map.fromList

text :: String -> Value Natural
text = Sequence . map (Atom . Token . fromIntegral . ord)

number :: Double -> Value Natural
number = Atom . Number . toRational

input :: Plan scope -> ByteString
input = planInput

program :: Plan scope -> ByteString
program = A.bytes . planChecked

emission :: Plan scope -> E.Emission
emission = planEmission

rollout :: Plan scope -> R.Batch scope
rollout = planBatch

invocation :: V.Binding -> Plan scope -> Either Error V.Runtime
invocation bound planned = do
    lifecycle (V.prepare (V.Selection (V.boundCall bound) (planWorld planned)) (V.start (planChecked planned) command))
  where
    command = 0
    lifecycle = either (Left . Lifecycle) Right
