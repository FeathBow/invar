{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}

module LearnerFixture (Exchange (..), Scenario (..), withPlan, prepare, scenario, run, execute, worker, owner, timer, wire, require) where

import BatchCalls (quote)
import Calls qualified
import Data.Aeson (Value (..), eitherDecodeStrict, object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Foldable (toList)
import Data.IORef (modifyIORef', newIORef, readIORef)
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8, encodeUtf8)
import Hedgehog
import Invar.Artifact qualified as Artifact
import Invar.Infer qualified as Infer
import Invar.Learn qualified as L
import Invar.Learn.Wire qualified as Wire
import Invar.Learn.Worker qualified as W
import Invar.Learn.Worker.Resident qualified as Resident
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as R
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as Load
import Policies qualified
import Probabilities qualified
import ResidentFixture (timer)
import ResidentWorkloads qualified as Rollout
import System.Directory (createDirectory)
import System.FilePath ((</>))
import Updates (change, field, stepRecords, wire)

data Exchange scope = Exchange {call :: W.Call scope, paths :: Resident.Paths, before :: [Value], steps :: [Value], after :: [Value], permission :: ByteString, release :: Value, released :: Value}
data Scenario scope = Scenario {groups :: [Exchange scope], closed :: Value, ending :: String}

require :: (Show problem) => Either problem value -> IO value
require = either (ioError . userError . show) pure

withPlan :: FilePath -> (forall scope. L.Plan scope -> IO value) -> PropertyT IO value
withPlan root action = do
    fixture <- Rollout.setup root 1 [Infer.artifact Calls.request]
    let selected = Rollout.initial fixture
        configured = L.Settings (Infer.artifact Calls.request) (replicate 64 'b') (Infer.artifact Calls.request) (Infer.tokenizer Calls.request) (replicate 64 '0') (replicate 64 '1') (Infer.base Calls.request) (Infer.assembly Calls.request) 0.2 0.04 0.0001 1 (L.Optimizer 0.002 0.8 0.95 0.0000001 0.01)
    observed <- evalIO $ R.withConfiguredDriver R.Resident (R.worker selected, R.sessions selected) $ \driver -> do
        batch <- R.run driver selected >>= require
        require (L.prepare configured batch) >>= action
    evalEither observed

owner :: Value
owner = object ["role" .= String "learning", "session" .= (0 :: Int)]

measurement :: Text.Text -> Text.Text
measurement = decodeUtf8 . wire . pure . timer

prepare :: FilePath -> (Int, V.Binding) -> L.Plan scope -> IO (Exchange scope)
prepare root (index, binding) planned = do
    call <- require (W.prepare binding planned)
    envelope <- require (eitherDecodeStrict (encodeUtf8 (Text.pack (W.input call))))
    image <- require (Wire.image (L.emission planned))
    let invocation = field "invocation" envelope
        bound = field "binding" invocation
        request = field "request" envelope
        load = field "load" envelope
        state = object [name .= field name request | name <- ["policy", "learner", "reference", "tokenizer", "base", "assembly", "optimizer"]]
        loaded = object ["stage" .= String "loaded_learner", "binding" .= bound, "load" .= load, "state" .= state, "image" .= object ["artifact" .= decodeUtf8 (Load.artifact image), "profile" .= decodeUtf8 (Load.profile image)], "model" .= String "host-protocol-fixture", "revision" .= String "fixture-v1"]
        consumed = object ["stage" .= String "consumed", "binding" .= bound, "load" .= load, "request" .= request, "program" .= field "program" invocation]
        directory = root </> ("update" ++ show index)
    createDirectory directory
    result <- artifacts directory request (loaded, consumed)
    records <- require (stepRecords bound request (field "adapter" result))
    let prefix = [timer "load" | index == 0] ++ [timer "activation", loaded, consumed]
        suffix = [timer "reward_update", result]
        transcript = directory </> "transcript.jsonl"
    Bytes.writeFile transcript (wire (prefix ++ records ++ suffix))
    digest <- Artifact.identity "Learner protocol fixture transcript" transcript
    let common = ["format" .= String "invar-resident-v1", "owner" .= owner, "loads" .= [load], "result_sha256" .= digest]
    pure (Exchange call (Resident.Paths (root </> "input checkpoint") directory) prefix records suffix (Bytes.init (wire [invocation])) (object ("action" .= String "release" : common)) (object (["stage" .= String "released", "measurement" .= measurement "released"] ++ common)))

artifacts :: FilePath -> Value -> (Value, Value) -> IO Value
artifacts directory request (loaded, consumed) = do
    let tensors = Policies.artifact [("fixture", [1], Bytes.pack ['\0', '\0', '\128', '\63'])]
    Bytes.writeFile (directory </> "adapter.safetensors") tensors
    Bytes.writeFile (directory </> "gradients.safetensors") tensors
    Bytes.writeFile (directory </> "learner.pt") "host protocol checkpoint bytes"
    policy <- Policy.identity (directory </> "adapter.safetensors")
    learner <- Artifact.identity "Learner fixture" (directory </> "learner.pt")
    gradients <- Artifact.identity "Gradient fixture" (directory </> "gradients.safetensors")
    let count = sum (map (length . array . field "behavior_bits") (array (field "samples" request)))
        update = object ["before" .= field "policy" request, "after" .= policy, "active_tokens" .= count, "nonzero_advantages" .= (0 :: Int), "gradient_norm" .= (0 :: Double), "reward_gradient_norm" .= (0 :: Double)]
        result = object ["stage" .= String "result", "binding" .= field "binding" consumed, "request" .= request, "update" .= update, "adapter" .= policy, "learner" .= learner, "gradients" .= gradients, "storage" .= String "staged; not published"]
        probability = directory </> "probabilities.json"
    Bytes.writeFile probability (Bytes.init (wire [Probabilities.fixture [loaded, consumed, result]]))
    digest <- Artifact.identity "Probability fixture" probability
    pure (change "probabilities" (toJSON digest) result)

array :: Value -> [Value]
array (Array values) = toList values
array _ = error "Expected learner fixture array"

scenario :: [Exchange scope] -> Scenario scope
scenario exchanges = Scenario exchanges (object ["stage" .= String "closed", "format" .= String "invar-resident-v1", "owner" .= owner, "groups" .= length exchanges, "measurement" .= measurement "closed"]) "IFS= read -r extra && exit 29\nexit 0"

worker :: FilePath -> W.Worker
worker root = W.Worker "/bin/sh" (root </> "learner.sh") root "initial launch checkpoint" (root </> "fixed reference") "initial launch output"

script :: FilePath -> Scenario scope -> String
script root selected = unlines (header ++ concat (zipWith groupScript [0 :: Int ..] (groups selected)) ++ closing)
  where
    header = ["test \"$1\" = " ++ quote ("--cache=" ++ root) ++ " || exit 20", "test \"$2\" = " ++ quote ("--reference=" ++ W.reference (worker root)) ++ " || exit 20", "test \"$3\" = '--session=0' || exit 20", "printf '%s\\n' \"$$\" >> " ++ quote (root </> "learner-pids")]
    closing = [receive "closing" (Bytes.init (wire [object ["format" .= String "invar-resident-v1", "owner" .= owner, "action" .= String "close"]])), "printf '%s\\n' closed > " ++ quote (root </> "learner-closed"), emit [closed selected], ending selected]
    groupScript index exchange =
        [ receive "request" (Bytes.init (wire [object ["format" .= String "invar-learning-resident-v1", "checkpoint" .= Resident.checkpoint (paths exchange), "output" .= Resident.output (paths exchange), "call" .= W.input (call exchange)]]))
        , emit (before exchange)
        , receive "permission" (permission exchange)
        , "printf '%s\\n' approved > " ++ quote (root </> ("learner-approved" ++ show index))
        ]
            ++ concatMap stepScript (steps exchange)
            ++ [ emit (after exchange)
               , receive "release" (Bytes.init (wire [release exchange]))
               , "printf '%s\\n' released > " ++ quote (root </> ("learner-released" ++ show index))
               , emit [released exchange]
               ]
    stepScript record = emit [record] : ["IFS= read -r reply || exit 23" | Just (String "current") <- [stage record]]
    stage (Object fields) = Fields.lookup "stage" fields
    stage _ = Nothing
    receive variable expected = "IFS= read -r " ++ variable ++ " || exit 21\ntest \"$" ++ variable ++ "\" = " ++ quote (Bytes.unpack expected) ++ " || exit 22"
    emit values = "printf '%s\\n' " ++ unwords (map (quote . Bytes.unpack) (Bytes.lines (wire values)))

run :: FilePath -> Scenario scope -> (forall ownerScope. Resident.Resident ownerScope -> IO (Either W.Failure value)) -> IO (Either W.Failure value, ByteString)
run root selected action = do
    writeFile (W.script (worker root)) (script root selected)
    buffer <- newIORef []
    returned <- Resident.withResident (Resident.Options (worker root) 0 (\line -> modifyIORef' buffer (line :))) action
    emitted <- Bytes.unlines . reverse <$> readIORef buffer
    pure (returned, emitted)

execute :: Resident.Resident ownerScope -> [Exchange scope] -> IO (Either W.Failure [Resident.Receipt scope])
execute _ [] = pure (Right [])
execute ownerScope (exchange : remaining) = do
    returned <- Resident.run ownerScope (paths exchange) (call exchange)
    case returned of
        Left problem -> pure (Left problem)
        Right receipt -> fmap (receipt :) <$> execute ownerScope remaining
