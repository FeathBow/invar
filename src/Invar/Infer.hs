{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

module Invar.Infer (Request (..), Plan, Error (..), prepare, bindPolicy, boundPolicy, emission, fromEmission, invocation, arguments, requested, image) where

import Control.Monad (unless, when)
import Data.ByteString (ByteString)
import Data.Char (chr, ord)
import Data.Map.Strict qualified as Map
import Data.Ratio (denominator, numerator)
import Data.Set qualified as Set
import Invar.Construct qualified as C
import Invar.Digest qualified as Digest
import Invar.Infer.Schema (Inputs)
import Invar.Infer.Schema qualified as Schema
import Invar.Materialization qualified as Materialization
import Invar.Policy.Description qualified as Policy
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as Load
import Invar.Spec.Program qualified as P
import Invar.Spec.Value (Scalar (..), Value (..))
import Numeric.Natural (Natural)

data Request = Request
    { artifact :: String
    , tokenizer :: String
    , base :: String
    , assembly :: String
    , prompt :: String
    , tokens :: Natural
    , temperature :: Double
    , seed :: Integer
    }
    deriving (Eq, Show)

data Plan = Plan Request (Maybe Policy.Description)
    deriving (Eq, Show)

data Error
    = InvalidRequest String
    | ConstructionError C.BuildError
    | EvaluationError E.Error
    | InvalidEmission String
    | InvocationError V.Error
    deriving (Eq, Show)

prepare :: Request -> Either Error Plan
prepare request = do
    selected <- emission request >>= fromEmission
    pure (Plan selected Nothing)

bindPolicy :: Policy.Description -> Plan -> Either Error Plan
bindPolicy selected planned = do
    let request = requested planned
    unless (Policy.bindings selected == (artifact request, tokenizer request, base request, assembly request)) (Left (InvalidRequest "Inference inputs differ from the selected policy description"))
    pure (Plan request (Just selected))

boundPolicy :: Plan -> Maybe Policy.Description
boundPolicy (Plan _ selected) = selected

fromEmission :: E.Emission -> Either Error Request
fromEmission (E.Emission "infer" "categorical-inference/v1" payload) = lower payload
fromEmission _ = Left (InvalidEmission "Expected one categorical-inference/v1 operation")

emission :: Request -> Either Error E.Emission
emission request = do
    validate request
    checked <- either (Left . ConstructionError) Right program
    emissions <- either (Left . EvaluationError) Right (A.run checked (world request))
    case emissions of
        [result@(E.Emission "infer" "categorical-inference/v1" _)] -> Right result
        _ -> Left (InvalidEmission "Expected one inference command")

invocation :: V.Binding -> Plan -> Either Error (ByteString, V.Runtime)
invocation bound planned = do
    checked <- either (Left . ConstructionError) Right program
    ready <- convert (V.prepare (V.Selection (V.boundCall bound) (world (requested planned))) (V.start checked 0))
    pure (A.bytes checked, ready)
  where
    convert = either (Left . InvocationError) Right

program :: Either C.BuildError A.Checked
program = C.compile meaning [C.emit @"infer" @"categorical-inference/v1" @'[ 'C.Semantic "request", 'C.Semantic "policy", 'C.LogicalRandom "sample"] expression]
  where
    expression = C.record (C.field @"policy" (C.source @('C.Semantic "policy") @(C.Record '[ '("artifact", [Natural]), '("profile", [Natural])])) (C.field @"request" (C.source @('C.Semantic "request") @Inputs) (C.field @"seed" (C.numberSource @('C.LogicalRandom "sample")) C.emptyFields)))
    inputs = Schema.inputs
    policy = Schema.policy
    output = P.RecordType (Map.fromList [("policy", policy), ("request", inputs), ("seed", P.NumberType)])
    sources = Map.fromList [(P.Semantic "request", inputs), (P.Semantic "policy", policy), (P.LogicalRandom "sample", P.NumberType)]
    sink = P.Sink "categorical-inference/v1" output (Map.keysSet sources) Set.empty
    meaning = E.Semantics (P.Schema sources Map.empty (Map.singleton "infer" sink)) Map.empty

world :: Request -> E.World
world request = Map.fromList [(P.Semantic "request", inputs), (P.Semantic "policy", Load.imageValue (image request)), (P.LogicalRandom "sample", Atom (Number (fromInteger (seed request))))]
  where
    inputs = Record (Map.fromList [("artifact", characters (artifact request)), ("tokenizer", characters (tokenizer request)), ("base", characters (base request)), ("assembly", characters (assembly request)), ("prompt", characters (prompt request)), ("tokens", Atom (Token (tokens request))), ("temperature", Atom (Number (toRational (temperature request))))])
    characters = Sequence . map (Atom . Token . fromIntegral . ord)

image :: Request -> Load.Image
image request = Materialization.image (artifact request, tokenizer request, base request, assembly request)

validate :: Request -> Either Error ()
validate request = do
    mapM_ identity [("adapter", artifact request), ("tokenizer", tokenizer request), ("frozen base", base request), ("model assembly", assembly request)]
    unless (tokens request > 0) (Left (InvalidRequest "Token limit must be positive"))
    let value = temperature request
    unless (not (isNaN value || isInfinite value) && value > 0) (Left (InvalidRequest "Temperature must be finite and positive"))
    when ('\0' `elem` prompt request) (Left (InvalidRequest "A process argument cannot contain NUL"))
  where
    identity (label, value) = unless (Digest.sha256 value) (Left (InvalidRequest ("Expected a lowercase SHA-256 " ++ label ++ " identity")))

lower :: Value Natural -> Either Error Request
lower payload = do
    inputs <- field "request" payload
    identity <- field "artifact" inputs >>= text
    encoding <- field "tokenizer" inputs >>= text
    frozen <- field "base" inputs >>= text
    configuration <- field "assembly" inputs >>= text
    input <- field "prompt" inputs >>= text
    cap <- field "tokens" inputs >>= natural
    thermal <- field "temperature" inputs >>= rational
    random <- field "seed" payload >>= rational
    unless (denominator random == 1) (Left (InvalidEmission "Sample seed must be integral"))
    let request = Request {artifact = identity, tokenizer = encoding, base = frozen, assembly = configuration, prompt = input, tokens = cap, temperature = fromRational thermal, seed = numerator random}
    supplied <- field "policy" payload
    unless (supplied == Load.imageValue (image request)) (Left (InvalidEmission "Policy materialization differs from the inference inputs"))
    pure request

field :: String -> Value Natural -> Either Error (Value Natural)
field name (Record fields) = maybe (Left (InvalidEmission ("Missing field: " ++ name))) Right (Map.lookup name fields)
field _ _ = Left (InvalidEmission "Expected a record")

text :: Value Natural -> Either Error String
text (Sequence values) = traverse character values
  where
    character (Atom (Token value))
        | value <= fromIntegral (ord (maxBound :: Char)) = Right (chr (fromIntegral value))
    character _ = Left (InvalidEmission "Expected a character code point")
text _ = Left (InvalidEmission "Expected a text sequence")

natural :: Value Natural -> Either Error Natural
natural (Atom (Token value)) = Right value
natural _ = Left (InvalidEmission "Expected a natural number")

rational :: Value Natural -> Either Error Rational
rational (Atom (Number value)) = Right value
rational _ = Left (InvalidEmission "Expected a rational number")

arguments :: Plan -> [String]
arguments (Plan request _) = ["--digest=" ++ artifact request, "--tokenizer-digest=" ++ tokenizer request, "--base-digest=" ++ base request, "--assembly-digest=" ++ assembly request, "--prompt=" ++ prompt request, "--tokens=" ++ show (tokens request), "--temperature=" ++ show (temperature request), "--seed=" ++ show (seed request)]

requested :: Plan -> Request
requested (Plan request _) = request
