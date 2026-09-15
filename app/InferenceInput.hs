module InferenceInput (request, requestFor, binding, options, requestWith, bindingWith, optionsWith, mode) where

import Invar.Infer qualified as Infer
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as Rollout
import Invar.Spec.Invocation qualified as V
import Options qualified as O
import System.Console.GetOpt (OptDescr)

mode :: Maybe String -> Either String Rollout.Mode
mode Nothing = Right Rollout.Serial
mode (Just "serial") = Right Rollout.Serial
mode (Just "batch") = Right Rollout.Batched
mode (Just "resident") = Right Rollout.Resident
mode (Just "shared") = Right Rollout.Shared
mode _ = Left "Invalid inference mode: expected serial, batch, resident or shared"

request :: O.Fields -> Either String Infer.Request
request = requestWith ""

requestWith :: String -> O.Fields -> Either String Infer.Request
requestWith prefix fields = do
    artifact <- string "digest"
    tokenizer <- string "tokenizer-digest"
    base <- string "base-digest"
    assembly <- string "assembly-digest"
    inputs prefix (artifact, tokenizer, base, assembly) fields
  where
    string key = O.required fields (prefix ++ key)

requestFor :: Policy.Description -> O.Fields -> Either String Infer.Request
requestFor selected = inputs "" (Policy.bindings selected)

inputs :: String -> (String, String, String, String) -> O.Fields -> Either String Infer.Request
inputs prefix (artifact, tokenizer, base, assembly) fields = do
    prompt <- string "prompt"
    tokens <- O.numeric fields (prefix ++ "tokens")
    temperature <- O.numeric fields (prefix ++ "temperature")
    seed <- O.numeric fields (prefix ++ "seed")
    pure Infer.Request {Infer.artifact = artifact, Infer.tokenizer = tokenizer, Infer.base = base, Infer.assembly = assembly, Infer.prompt = prompt, Infer.tokens = tokens, Infer.temperature = temperature, Infer.seed = seed}
  where
    string key = O.required fields (prefix ++ key)

binding :: O.Fields -> Either String V.Binding
binding = bindingWith ""

bindingWith :: String -> O.Fields -> Either String V.Binding
bindingWith prefix fields = V.Binding . V.CallId <$> O.numeric fields (prefix ++ "call") <*> (V.AttemptId <$> O.numeric fields (prefix ++ "attempt")) <*> (V.Instance <$> O.numeric fields (prefix ++ "instance"))

options :: [OptDescr (String, String)]
options = optionsWith ""

optionsWith :: String -> [OptDescr (String, String)]
optionsWith prefix = O.descriptions [(prefix ++ key, description) | (key, description) <- [("digest", "Canonical adapter tensor SHA-256"), ("tokenizer-digest", "Tokenizer operation SHA-256"), ("base-digest", "Frozen model tensor SHA-256"), ("assembly-digest", "Model assembly SHA-256"), ("prompt", "Input prompt"), ("tokens", "Positive token limit"), ("temperature", "Positive sampling temperature"), ("seed", "Logical sample seed"), ("call", "Logical call identity"), ("attempt", "Dispatch attempt identity"), ("instance", "Executor load-instance identity")]]
