{-# LANGUAGE RoleAnnotations #-}

module Invar.Learn (Optimizer (..), Settings (..), Plan, Error (..), validate, materialization, prepare, input, program, emission, rollout, invocation) where

import Control.Monad (unless)
import Data.Aeson (encode)
import Data.ByteString (ByteString)
import Data.ByteString.Lazy qualified as Lazy
import Data.Char (ord)
import Data.List (sortOn)
import Data.Map.Strict qualified as Map
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

data Settings = Settings {policy :: String, learner :: String, reference :: String, tokenizer :: String, base :: String, assembly :: String, clip :: Double, penalty :: Double, delta :: Double, optimizer :: Optimizer}
    deriving (Eq, Show)

type role Plan nominal
data Plan scope = Plan {planBatch :: R.Batch scope, planChecked :: A.Checked, planWorld :: E.World, planEmission :: E.Emission, planInput :: ByteString}

data Error = InvalidSettings String | PolicyMismatch | TokenizerMismatch | MaterializationMismatch | Construction C.BuildError | Evaluation E.Error | Lowering Wire.Error | Lifecycle V.Error | InvalidEmission
    deriving (Eq, Show)

prepare :: Settings -> R.Batch scope -> Either Error (Plan scope)
prepare settings batch = do
    validate settings
    let samples = R.samples batch
    unless (all ((== policy settings) . I.artifact . Result.consumed . R.observation) samples) (Left PolicyMismatch)
    mapM_ (materialization settings . Result.consumed . R.observation) samples
    checked <- either (Left . Construction) Right P.checked
    let sources = world settings samples
    commands <- either (Left . Evaluation) Right (A.run checked sources)
    case commands of
        [command] -> do
            payload <- either (Left . Lowering) Right (Wire.lower command)
            pure Plan {planBatch = batch, planChecked = checked, planWorld = sources, planEmission = command, planInput = Lazy.toStrict (encode payload)}
        _ -> Left InvalidEmission

validate :: Settings -> Either Error ()
validate settings = do
    unless (all identity [policy settings, learner settings, reference settings, base settings, assembly settings]) (Left (InvalidSettings "Expected lowercase SHA-256 artifact identities"))
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
    unless (base settings == I.base requested && assembly settings == I.assembly requested) (Left MaterializationMismatch)

world :: Settings -> [R.Sample] -> E.World
world settings samples = Map.fromList [(Semantic "policy", Load.imageValue image), (Semantic "learner", learnerValue settings), (Semantic "reference", text (reference settings)), (Semantic "algorithm", algorithm), (Semantic "trajectories", keyed (trajectory . R.observation)), (Semantic "behavior", keyed (Sequence . map (Atom . Bits32) . Result.behaviorBits . R.observation)), (Semantic "rewards", keyed (Atom . Number . R.reward)), (Semantic "groups", groupValues indexed), (Semantic "order", Sequence [Mapping (Map.singleton index marker) | (index, _) <- indexed])]
  where
    image = Materialization.learning (policy settings, learner settings, tokenizer settings, base settings, assembly settings, reference settings)
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

groupValues :: [(Natural, R.Sample)] -> Value Natural
groupValues samples = Sequence [Mapping members | (_, members) <- sortOn fst (Map.elems grouped)]
  where
    grouped = foldl' insert Map.empty samples
    insert groups (index, sample) = Map.insert (R.group sample) updated groups
      where
        updated = case Map.lookup (R.group sample) groups of
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
