{-# LANGUAGE RoleAnnotations #-}

module Invar.Learn (Optimizer (..), Schedule (..), ReferenceSource (..), Algorithm (..), Settings (..), Reading (..), PlanOf, Plan, Error (..), synchronous, validate, materialization, prepare, distilled, observedInput, input, program, emission, rollout, invocation) where

import Control.Monad (unless)
import Data.Aeson (encode)
import Data.ByteString (ByteString)
import Data.ByteString.Lazy qualified as Lazy
import Data.Char (ord)
import Data.List (sortOn)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text qualified as Text
import Data.Word (Word32)
import Invar.Cohort qualified as Cohort
import Invar.Construct qualified as C
import Invar.Digest qualified as Digest
import Invar.Infer qualified as I
import Invar.Infer.Output qualified as Output
import Invar.Infer.Result qualified as Result
import Invar.Infer.Trajectory (Trajectory)
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Learn.Algorithm (Optimizer (..))
import Invar.Learn.Algorithm qualified as Algorithm
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
    , algorithm :: Algorithm
    , steps :: Natural
    , optimizer :: Optimizer
    , schedule :: Schedule
    , referenceSource :: ReferenceSource
    }
    deriving (Eq, Show)

data Algorithm = Grpo Double | Opd Algorithm.Teacher
    deriving (Eq, Show)

data InputSample = InputSample {sampleGroup :: String, sampleTrajectory :: Trajectory, sampleReward :: Rational}

data Reading = Reading {readingCall :: V.Binding, readingWords :: [Word32]}
    deriving (Eq, Show)

type role PlanOf nominal nominal
data PlanOf scope grade = Plan {planBatch :: R.BatchOf scope grade, planChecked :: A.Checked, planWorld :: E.World, planEmission :: E.Emission, planInput :: ByteString}

type Plan scope = PlanOf scope Cohort.Graded

data Error = InvalidSettings String | PolicyMismatch | ReferenceMismatch | TokenizerMismatch | MaterializationMismatch | UnmatchedReadings | Construction C.BuildError | Evaluation E.Error | Lowering Wire.Error | Lifecycle V.Error | InvalidEmission
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

distilled :: Settings -> Map.Map String Reading -> R.BatchOf scope Cohort.Ungraded -> Either Error (PlanOf scope Cohort.Ungraded)
distilled settings readings batch = do
    validate settings
    chosen <- case algorithm settings of
        Opd teacher -> Right teacher
        Grpo _ -> Left (InvalidSettings "A distillation update needs distillation settings")
    unless (Map.keysSet readings == Set.fromList (map R.name (R.samples batch))) (Left UnmatchedReadings)
    let trajectories = map R.trajectory (R.samples batch)
        read' = [readings Map.! R.name sample | sample <- R.samples batch]
        keyed project = Mapping (Map.fromList (zip [0 ..] (map project read')))
        coefficients = record [("epsilon", number (clip settings)), ("penalty", number (penalty settings)), ("steps", Atom (Number (fromIntegral (steps settings))))]
        own = Map.fromList [(Semantic "algorithm", coefficients), (Semantic "teacher", teacherValue chosen), (Semantic "teacher_scores", keyed (Sequence . map (Atom . Bits32) . readingWords)), (Semantic "teacher_calls", keyed (bindingValue . readingCall))]
    admitted settings trajectories
    (checked, sources, command, payload) <- emitted P.distilled (Map.union own (sharedWorld settings trajectories))
    pure Plan {planBatch = batch, planChecked = checked, planWorld = sources, planEmission = command, planInput = payload}

compileInput :: Settings -> [InputSample] -> Either Error (A.Checked, E.World, E.Emission, ByteString)
compileInput settings samples = do
    validate settings
    chosenDelta <- case algorithm settings of
        Grpo value -> Right value
        Opd _ -> Left (InvalidSettings "A GRPO update needs GRPO settings")
    admitted settings (map sampleTrajectory samples)
    let indexed = zip [0 ..] samples
        keyed project = Mapping (Map.fromList [(index, project sample) | (index, sample) <- indexed])
        coefficients = record [("epsilon", number (clip settings)), ("penalty", number (penalty settings)), ("delta", number chosenDelta), ("steps", Atom (Number (fromIntegral (steps settings))))]
        own = Map.fromList [(Semantic "algorithm", coefficients), (Semantic "rewards", keyed (Atom . Number . sampleReward)), (Semantic "groups", groupValues indexed)]
    emitted P.checked (Map.union own (sharedWorld settings (map sampleTrajectory samples)))

admitted :: Settings -> [Trajectory] -> Either Error ()
admitted settings trajectories = do
    unless (all ((== behaviorPolicy (schedule settings)) . I.artifact . Trajectory.request) trajectories) (Left PolicyMismatch)
    mapM_ (materialization settings . Trajectory.request) trajectories
    unless (all (scoredBy settings . Trajectory.reference) trajectories) (Left ReferenceMismatch)

emitted :: Either C.BuildError A.Checked -> E.World -> Either Error (A.Checked, E.World, E.Emission, ByteString)
emitted declared sources = do
    checked <- either (Left . Construction) Right declared
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
    case algorithm settings of
        Grpo chosenDelta -> unless (Algorithm.coefficients (clip settings) (penalty settings) chosenDelta) (Left (InvalidSettings "Invalid GRPO coefficients"))
        Opd teacher -> do
            unless (Algorithm.clipping (clip settings) && Algorithm.penalty (penalty settings)) (Left (InvalidSettings "Invalid distillation coefficients"))
            either (Left . InvalidSettings) Right (Algorithm.teacher (tokenizer settings) teacher)
    unless (steps settings > 0) (Left (InvalidSettings "An update needs at least one optimizer step"))
    unless (identity (behaviorPolicy chosenSchedule)) (Left (InvalidSettings "Expected a lowercase SHA-256 behavior policy identity"))
    unless (version chosenSchedule == (if update chosenSchedule > staleness chosenSchedule then update chosenSchedule - staleness chosenSchedule else 0)) (Left (InvalidSettings "Samples must come from version max(0, update - staleness)"))
    unless (version chosenSchedule /= update chosenSchedule || behaviorPolicy chosenSchedule == policy settings) (Left (InvalidSettings "Samples of the update's own version must come from the policy being updated"))
    unless (Algorithm.optimizer (optimizer settings)) (Left (InvalidSettings "Invalid AdamW coefficients"))
  where
    chosenSchedule = schedule settings
    identity = Digest.sha256

materialization :: Settings -> I.Request -> Either Error ()
materialization settings requested = do
    unless (tokenizer settings == I.tokenizer requested) (Left TokenizerMismatch)
    unless (behaviorBase settings == I.base requested && behaviorAssembly settings == I.assembly requested) (Left MaterializationMismatch)

sharedWorld :: Settings -> [Trajectory] -> E.World
sharedWorld settings trajectories = Map.fromList [(Semantic "policy", Load.imageValue image), (Semantic "learner", learnerValue settings), (Semantic "reference", text (reference settings)), (Semantic "reference_source", text (Text.unpack (S.sourceName (referenceSource settings)))), (Semantic "trajectories", keyed (trajectory . Trajectory.result)), (Semantic "behavior_model", behaviorModel), (Semantic "schedule", record [("update", Atom (Token (update chosenSchedule))), ("staleness", Atom (Token (staleness chosenSchedule)))]), (Semantic "generations", keyed (const generation)), (Semantic "behavior", keyed (Sequence . map (Atom . Bits32) . Trajectory.behaviorBits)), (Semantic "reference_scores", keyed (Sequence . map (Atom . Bits32) . maybe [] Output.scores . Trajectory.reference)), (Semantic "order", Sequence [Mapping (Map.singleton index marker) | (index, _) <- indexed])]
  where
    image = Materialization.learning (policy settings, learner settings, tokenizer settings, base settings, assembly settings, reference settings)
    behaviorModel = record [("base", text (behaviorBase settings)), ("assembly", text (behaviorAssembly settings))]
    chosenSchedule = schedule settings
    generation = record [("version", Atom (Token (version chosenSchedule))), ("policy", text (behaviorPolicy chosenSchedule))]
    indexed = zip [0 :: Natural ..] trajectories
    keyed project = Mapping (Map.fromList [(index, project sample) | (index, sample) <- indexed])

teacherValue :: Algorithm.Teacher -> Value Natural
teacherValue chosen = record (("temperature", number (Algorithm.teacherTemperature chosen)) : [(name, text (field chosen)) | (name, field) <- [("model", Algorithm.teacherModel), ("revision", Algorithm.teacherRevision), ("adapter", Algorithm.teacherAdapter), ("tokenizer", Algorithm.teacherTokenizer), ("base", Algorithm.teacherBase), ("assembly", Algorithm.teacherAssembly), ("profile", Algorithm.teacherProfile), ("truncation", Algorithm.teacherTruncation), ("logprobs", Algorithm.teacherLogprobs), ("path", Algorithm.teacherPath)]])

bindingValue :: V.Binding -> Value Natural
bindingValue (V.Binding (V.CallId call) (V.AttemptId attempt) (V.Instance instance')) = record [("call", Atom (Token call)), ("attempt", Atom (Token attempt)), ("instance", Atom (Token instance'))]

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

input :: PlanOf scope grade -> ByteString
input = planInput

program :: PlanOf scope grade -> ByteString
program = A.bytes . planChecked

emission :: PlanOf scope grade -> E.Emission
emission = planEmission

rollout :: PlanOf scope grade -> R.BatchOf scope grade
rollout = planBatch

invocation :: V.Binding -> PlanOf scope grade -> Either Error V.Runtime
invocation bound planned = do
    lifecycle (V.prepare (V.Selection (V.boundCall bound) (planWorld planned)) (V.start (planChecked planned) command))
  where
    command = 0
    lifecycle = either (Left . Lifecycle) Right
