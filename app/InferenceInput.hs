module InferenceInput (request, requestFor, binding, options, requestWith, bindingWith, optionsWith, declaredWith, declarationWith, calls, readCalls, mode) where

import Control.Monad (when, (>=>))
import Data.Aeson (eitherDecodeStrict)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.Maybe (isJust)
import Invar.Infer qualified as Infer
import Invar.Infer.Invocation qualified as Call
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as Rollout
import Invar.Spec.Invocation qualified as V
import Options qualified as O
import System.Console.GetOpt (OptDescr)
import System.Exit (die)
import System.FilePath ((</>))

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

declaredWith :: String -> O.Fields -> IO Infer.Plan
declaredWith prefix fields = case O.optional fields (prefix ++ "checkpoint") of
    Nothing -> either die pure (requestWith prefix fields >>= first show . Infer.prepare)
    Just checkpoint -> do
        when (any (isJust . O.optional fields . (prefix ++)) ["digest", "tokenizer-digest", "base-digest", "assembly-digest"]) (die ("--" ++ prefix ++ "checkpoint derives the adapter and materialization digests from policy.json; separate digest options are invalid"))
        selected <- Policy.readDescription (checkpoint </> "policy.json")
        either die pure (inputs prefix (Policy.bindings selected) fields >>= first show . (Infer.prepare >=> Infer.bindPolicy selected))

declarationWith :: String -> [OptDescr (String, String)]
declarationWith prefix = optionsWith prefix ++ O.descriptions [(prefix ++ "checkpoint", "Checkpoint whose policy.json declared the inference, in place of the four digests")]

calls :: ByteString -> Either String [Call.Call]
calls encoded = do
    arguments <- eitherDecodeStrict encoded
    when (null arguments) (Left "A finite inference batch requires at least one call")
    traverse call arguments
  where
    call supplied = do
        fields <- O.parse options supplied
        requested <- request fields
        planned <- first show (Infer.prepare requested)
        bound <- binding fields
        first show (Call.prepare bound planned)

readCalls :: FilePath -> IO [Call.Call]
readCalls path = Bytes.readFile path >>= either die pure . calls
