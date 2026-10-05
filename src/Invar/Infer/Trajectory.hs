{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Trajectory (
    Trajectory,
    Constraint (..),
    binding,
    request,
    constraint,
    model,
    revision,
    loaded,
    promptTokens,
    responseTokens,
    tokens,
    promptLength,
    behaviorBits,
    behavior,
    reference,
    truncated,
    text,
    result,
    evidence,
    digest,
) where

import Crypto.Hash.SHA256 qualified as SHA256
import Data.ByteString (ByteString)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8Lenient)
import Data.Word (Word32)
import GHC.Float (castDoubleToWord64)
import Invar.Artifact qualified as Artifact
import Invar.Canonical qualified as Canonical
import Invar.Infer qualified as I
import Invar.Infer.Output qualified as Output
import Invar.Infer.Result qualified as R
import Invar.Infer.Trajectory.Internal (Constraint (..), Trajectory)
import Invar.Infer.Trajectory.Internal qualified as Internal
import Invar.Policy qualified as Policy
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as L
import Numeric.Natural (Natural)

binding :: Trajectory -> V.Binding
binding = V.completedBinding . Internal.completion

request :: Trajectory -> I.Request
request = R.consumed . result

constraint :: Trajectory -> Constraint
constraint = Internal.constraint

model :: Trajectory -> String
model = Internal.model . Internal.observed

revision :: Trajectory -> String
revision = Internal.revision . Internal.observed

loaded :: Trajectory -> L.Fact
loaded = Internal.fact . Internal.observed

result :: Trajectory -> R.Result
result = Internal.result

promptTokens :: Trajectory -> [Natural]
promptTokens selected = take (fromIntegral (promptLength selected)) (tokens selected)

responseTokens :: Trajectory -> [Natural]
responseTokens selected = drop (fromIntegral (promptLength selected)) (tokens selected)

tokens :: Trajectory -> [Natural]
tokens = R.tokens . result

promptLength :: Trajectory -> Natural
promptLength = R.promptLength . result

behaviorBits :: Trajectory -> [Word32]
behaviorBits = R.behaviorBits . result

behavior :: Trajectory -> [Double]
behavior = R.behavior . result

reference :: Trajectory -> Maybe Output.Scored
reference = R.referenceScores . result

truncated :: Trajectory -> Bool
truncated = R.truncated . result

text :: Trajectory -> String
text = R.response . result

evidence :: Trajectory -> ByteString
evidence selected =
    Canonical.encode $
        object
            [ ("format", Canonical.Text "invar-trajectory-evidence-v1")
            , ("call", object [("binding", bindingValue (binding selected)), ("program", bytes (V.completedProgram (Internal.completion selected))), ("load", object [("binding", bindingValue (V.completedBinding loading)), ("program", bytes (V.completedProgram loading))])])
            , ("declared", declaredValue (constraint selected) (request selected))
            , ("request", object [("prompt", string (I.prompt submitted)), ("tokens", natural (I.tokens submitted)), ("seed", Canonical.Integer (I.seed submitted)), ("temperature", Canonical.Integer (toInteger (castDoubleToWord64 (I.temperature submitted))))])
            , ("observed", observedValue)
            ]
  where
    submitted = request selected
    seen = Internal.observed selected
    loading = L.report (Internal.fact seen)
    image = L.image (L.description (Internal.fact seen))
    observedValue =
        object
            [ ("requested", string (Internal.requested seen))
            , ("consumed", string (Internal.consumed seen))
            , ("tokenizer", string (Internal.tokenizer seen))
            , ("base", string (Internal.base seen))
            , ("assembly", string (Internal.assembly seen))
            , ("model", string (Internal.model seen))
            , ("revision", string (Internal.revision seen))
            , ("image", object [("artifact", bytes (L.artifact image)), ("profile", bytes (L.profile image))])
            , ("prompt_tokens", Canonical.Array (map natural (promptTokens selected)))
            , ("response_tokens", Canonical.Array (map natural (responseTokens selected)))
            , ("behavior_bits", Canonical.Array (map (Canonical.Integer . toInteger) (behaviorBits selected)))
            , ("reference", maybe Canonical.Null scored (reference selected))
            , ("text", string (text selected))
            , ("truncated", Canonical.Boolean (truncated selected))
            ]
    scored chosen = object [("adapter", string (Output.adapter chosen)), ("bits", Canonical.Array (map (Canonical.Integer . toInteger) (Output.scores chosen)))]

digest :: Trajectory -> String
digest = Artifact.hex . SHA256.hash . evidence

declaredValue :: Constraint -> I.Request -> Canonical.Value
declaredValue chosen submitted = case chosen of
    Materialization -> object (("kind", Canonical.Text "materialization") : identities)
    Described selected -> object ([("kind", Canonical.Text "policy"), ("model", string (Policy.model selected)), ("revision", string (Policy.revision selected))] ++ identities)
  where
    identities = [("adapter", string (I.artifact submitted)), ("tokenizer", string (I.tokenizer submitted)), ("base", string (I.base submitted)), ("assembly", string (I.assembly submitted))]

bindingValue :: V.Binding -> Canonical.Value
bindingValue (V.Binding (V.CallId call) (V.AttemptId attempt) (V.Instance instanceId)) = object [("call", natural call), ("attempt", natural attempt), ("instance", natural instanceId)]

object :: [(Text, Canonical.Value)] -> Canonical.Value
object = Canonical.Object . Map.fromList

string :: String -> Canonical.Value
string = Canonical.Text . Text.pack

natural :: Natural -> Canonical.Value
natural = Canonical.Integer . toInteger

bytes :: ByteString -> Canonical.Value
bytes = Canonical.Text . decodeUtf8Lenient
