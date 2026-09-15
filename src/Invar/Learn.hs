{-# LANGUAGE RoleAnnotations #-}

module Invar.Learn (Optimizer (..), Settings (..), Plan, Error (..), validate, materialization, prepare, observedInput, input, program, emission, rollout, invocation) where

import Control.Monad (unless)
import Data.Aeson (encode)
import Data.ByteString (ByteString)
import Data.ByteString.Lazy qualified as Lazy
import Data.Char (ord)
import Data.List (sortOn)
import Data.Map.Strict qualified as Map
import Invar.Cohort qualified as Cohort
import Invar.Construct qualified as C
import Invar.Infer qualified as I
import Invar.Infer.Result qualified as Result
import Invar.Learn.Program qualified as P
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
    , optimizer :: Optimizer
    }
    deriving (Eq, Show)

data InputSample = InputSample {sampleGroup :: String, sampleResult :: Result.Result, sampleReward :: Rational}

type role Plan nominal
data Plan scope = Plan {planBatch :: R.Batch scope, planChecked :: A.Checked, planWorld :: E.World, planEmission :: E.Emission, planInput :: ByteString}

data Error = InvalidSettings String | PolicyMismatch | TokenizerMismatch | MaterializationMismatch | Construction C.BuildError | Evaluation E.Error | Lowering Wire.Error | Lifecycle V.Error | InvalidEmission
    deriving (Eq, Show)

prepare :: Settings -> R.Batch scope -> Either Error (Plan scope)
prepare settings batch = do
    let samples = [InputSample (R.group sample) (R.observation sample) (R.reward sample) | sample <- R.samples batch]
    (checked, sources, command, payload) <- compileInput settings samples
    pure Plan {planBatch = batch, planChecked = checked, planWorld = sources, planEmission = command, planInput = payload}

-- Reconstruct the numerical input from an admitted cohort observation. This
-- returns program/input bytes, without an invocation or execution permission.
observedInput :: Settings -> Cohort.Batch scope -> Either Error (ByteString, ByteString)
observedInput settings batch = do
    let samples = [InputSample (Cohort.group (Cohort.source sample)) (Cohort.observed sample) (Cohort.reward sample) | sample <- Cohort.observations batch]
    (checked, _, _, payload) <- compileInput settings samples
    pure (A.bytes checked, payload)

compileInput :: Settings -> [InputSample] -> Either Error (A.Checked, E.World, E.Emission, ByteString)
compileInput settings samples = do
    validate settings
    unless (all ((== policy settings) . I.artifact . Result.consumed . sampleResult) samples) (Left PolicyMismatch)
    mapM_ (materialization settings . Result.consumed . sampleResult) samples
    checked <- either (Left . Construction) Right P.checked
    let sources = world settings samples
    commands <- either (Left . Evaluation) Right (A.run checked sources)
    case commands of
        [command] -> do
            payload <- either (Left . Lowering) Right (Wire.lower command)
            pure (checked, sources, command, Lazy.toStrict (encode payload))
        _ -> Left InvalidEmission

validate :: Settings -> Either Error ()
validate settings = do
    unless (all identity [policy settings, learner settings, reference settings, base settings, assembly settings, behaviorBase settings, behaviorAssembly settings]) (Left (InvalidSettings "Expected lowercase SHA-256 artifact identities"))
    unless (identity (tokenizer settings)) (Left (InvalidSettings "Expected a lowercase SHA-256 tokenizer identity"))
    unless algorithm (Left (InvalidSettings "Invalid GRPO coefficients"))
    unless adamw (Left (InvalidSettings "Invalid AdamW coefficients"))
  where
    algorithm = finite (clip settings) && clip settings > 0 && clip settings < 1 && nonnegative (penalty settings) && positive (delta settings)
    adamw = nonnegative (learningRate chosen) && positive (epsilon chosen) && nonnegative (weightDecay chosen) && all moment [firstMoment chosen, secondMoment chosen]
    chosen = optimizer settings
    identity value = length value == digestLength && all (`elem` (['0' .. '9'] ++ ['a' .. 'f'])) value
    digestLength = 64
    finite value = not (isNaN value || isInfinite value)
    positive value = finite value && value > 0
    nonnegative value = finite value && value >= 0
    moment value = nonnegative value && value < 1

materialization :: Settings -> I.Request -> Either Error ()
materialization settings requested = do
    unless (tokenizer settings == I.tokenizer requested) (Left TokenizerMismatch)
    unless (behaviorBase settings == I.base requested && behaviorAssembly settings == I.assembly requested) (Left MaterializationMismatch)

world :: Settings -> [InputSample] -> E.World
world settings samples = Map.fromList [(Semantic "policy", Load.imageValue image), (Semantic "learner", learnerValue settings), (Semantic "reference", text (reference settings)), (Semantic "algorithm", algorithm), (Semantic "trajectories", keyed (trajectory . sampleResult)), (Semantic "behavior_model", behaviorModel), (Semantic "behavior", keyed (Sequence . map (Atom . Bits32) . Result.behaviorBits . sampleResult)), (Semantic "rewards", keyed (Atom . Number . sampleReward)), (Semantic "groups", groupValues indexed), (Semantic "order", Sequence [Mapping (Map.singleton index marker) | (index, _) <- indexed])]
  where
    image = Materialization.learning (policy settings, learner settings, tokenizer settings, base settings, assembly settings, reference settings)
    behaviorModel = record [("base", text (behaviorBase settings)), ("assembly", text (behaviorAssembly settings))]
    indexed = zip [0 ..] samples
    keyed project = Mapping (Map.fromList [(index, project sample) | (index, sample) <- indexed])
    algorithm = record [("epsilon", number (clip settings)), ("penalty", number (penalty settings)), ("delta", number (delta settings))]

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
