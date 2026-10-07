{-# LANGUAGE OverloadedStrings #-}

module Invar.Async.Entry (Entry (..), Claim (..), Role (..), Generation (..), claim, encode, decode, claimValue, claimed, roleName, roleOf, outcomeFields, outcome) where

import Data.Aeson (Object, Value (..), object, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Pair, Parser)
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8, encodeUtf8)
import Invar.Async.Completion qualified as Completion
import Invar.Async.Core (Attempt (..), Epoch (..), Worker (..))
import Invar.Async.Core qualified as Core
import Invar.Async.Plan (Request (..), Update (..))
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Policy qualified as Policy
import Invar.Spec.Invocation qualified as V
import Invar.Transcript (Outcome (..), Output (..))
import Numeric.Natural (Natural)
import System.Exit (ExitCode (..))

data Role = Inference | Learner
    deriving (Eq, Show)

data Claim
    = Connected Natural Natural
    | Lost Natural Natural
    | Started Natural Natural Natural
    | Completed Natural Natural Natural String
    | Ready Natural Natural V.Binding String String
    | Current Natural Natural Natural String
    | Applied Natural Natural V.Binding String Natural String String String
    | Staged Natural Natural String
    | Recorded Natural Natural
    | Committed Natural Natural
    | Abandoned Natural Natural
    deriving (Eq, Show)

data Generation = Generation {version :: Natural, adapter :: String, learner :: String, description :: Policy.Description}
    deriving (Eq, Show)

data Entry
    = Declared FilePath Object
    | Opened Natural Natural
    | Dispatched Natural Natural Natural V.Binding Natural
    | Attempted Natural Natural V.Binding Natural
    | Reserved Natural Role Natural Natural
    | Finished Natural Outcome
    | Happened Claim
    | Stored Natural V.Binding String
    | Verified Natural Natural String
    | Restarted [Generation]
    | Elapsed Text Natural Double Double
    deriving (Eq, Show)

format :: Text
format = "invar-runtime-journal-v2"

claim :: Core.Event -> Claim
claim event = case event of
    Core.Connected (Worker worker) (Epoch used) -> Connected worker used
    Core.Lost (Worker worker) (Epoch used) -> Lost worker used
    Core.Started (Worker worker) (Epoch used) (Request request) -> Started worker used request
    Core.Completed (Worker worker) (Epoch used) (Request request) digest -> Completed worker used request digest
    Core.Ready (Update update) (Attempt attempt) bound identity before -> Ready update attempt bound identity before
    Core.Current (Update update) (Attempt attempt) index before -> Current update attempt index before
    Core.Applied (Update update) (Attempt attempt) done -> Applied update attempt (Completion.binding done) (Completion.plan done) (Completion.step done) (Completion.consumed done) (Completion.before done) (Completion.after done)
    Core.Staged (Update update) (Attempt attempt) digest -> Staged update attempt digest
    Core.Recorded (Update update) (Attempt attempt) -> Recorded update attempt
    Core.Committed (Update update) (Attempt attempt) -> Committed update attempt
    Core.Abandoned (Update update) (Attempt attempt) -> Abandoned update attempt

encode :: Entry -> Value
encode recorded = case recorded of
    Declared directory fields -> Object (Fields.insert "entry" "declaration" (Fields.insert "format" (String format) (Fields.insert "directory" (String (Text.pack directory)) fields)))
    Opened worker used -> entry "epoch" ["worker" .= worker, "epoch" .= used]
    Dispatched request worker used bound process -> entry "dispatched" ["request" .= request, "worker" .= worker, "epoch" .= used, "binding" .= Wire.bindingValue bound, "process" .= process]
    Attempted update attempt bound process -> entry "attempt" ["update" .= update, "attempt" .= attempt, "binding" .= Wire.bindingValue bound, "process" .= process]
    Reserved number chosen slot used -> entry "process" ["process" .= number, "role" .= roleName chosen, "slot" .= slot, "epoch" .= used]
    Finished number ended -> entry "exit" (("process" .= number) : outcomeFields ended)
    Happened event -> entry "event" ["event" .= claimValue event]
    Stored request bound digest -> entry "result" ["request" .= request, "binding" .= Wire.bindingValue bound, "digest" .= digest]
    Verified update attempt digest -> entry "checkpoint" ["update" .= update, "attempt" .= attempt, "learner" .= digest]
    Restarted observed -> entry "restart" ["generations" .= map generationValue observed]
    Elapsed kind update started ended -> entry "interval" ["role" .= kind, "update" .= update, "start" .= started, "end" .= ended]
  where
    entry name fields = object (("entry" .= (name :: Text)) : fields)

decode :: Object -> Parser Entry
decode fields = do
    kind <- fields .: "entry"
    let exactly keys = Json.fields ("entry" : keys) fields
    case kind :: Text of
        "declaration" -> do
            declared <- fields .: "format"
            if declared == format then Declared <$> fields .: "directory" <*> pure (foldr Fields.delete fields ["entry", "format", "directory"]) else fail "The journal declares another format"
        "epoch" -> exactly ["worker", "epoch"] >> Opened <$> fields .: "worker" <*> fields .: "epoch"
        "dispatched" -> exactly ["request", "worker", "epoch", "binding", "process"] >> Dispatched <$> fields .: "request" <*> fields .: "worker" <*> fields .: "epoch" <*> binding fields <*> fields .: "process"
        "attempt" -> exactly ["update", "attempt", "binding", "process"] >> Attempted <$> fields .: "update" <*> fields .: "attempt" <*> binding fields <*> fields .: "process"
        "process" -> exactly ["process", "role", "slot", "epoch"] >> Reserved <$> fields .: "process" <*> (fields .: "role" >>= roleOf) <*> fields .: "slot" <*> fields .: "epoch"
        "exit" -> Finished <$> fields .: "process" <*> outcome fields
        "event" -> exactly ["event"] >> Happened <$> (fields .: "event" >>= withObject "journaled event" claimed)
        "result" -> exactly ["request", "binding", "digest"] >> Stored <$> fields .: "request" <*> binding fields <*> fields .: "digest"
        "checkpoint" -> exactly ["update", "attempt", "learner"] >> Verified <$> fields .: "update" <*> fields .: "attempt" <*> fields .: "learner"
        "restart" -> exactly ["generations"] >> Restarted <$> (fields .: "generations" >>= traverse (withObject "observed generation" generation))
        "interval" -> exactly ["role", "update", "start", "end"] >> Elapsed <$> fields .: "role" <*> fields .: "update" <*> fields .: "start" <*> fields .: "end"
        _ -> fail ("Unknown journal entry: " ++ show kind)

binding :: Object -> Parser V.Binding
binding fields = do
    fields .: "binding" >>= withObject "journaled binding" (Json.fields ["call", "attempt", "instance"])
    Wire.binding fields

generationValue :: Generation -> Value
generationValue observed = object ["version" .= version observed, "adapter" .= adapter observed, "learner" .= learner observed, "description" .= decodeUtf8 (Policy.encodeDescription (description observed))]

generation :: Object -> Parser Generation
generation fields = do
    Json.fields ["version", "adapter", "learner", "description"] fields
    described <- fields .: "description" >>= either fail pure . Policy.decodeDescription . encodeUtf8
    Generation <$> fields .: "version" <*> fields .: "adapter" <*> fields .: "learner" <*> pure described

roleName :: Role -> Text
roleName Inference = "inference"
roleName Learner = "learner"

roleOf :: Text -> Parser Role
roleOf "inference" = pure Inference
roleOf "learner" = pure Learner
roleOf other = fail ("Unknown process role: " ++ show other)

outcomeFields :: Outcome -> [Pair]
outcomeFields chosen = case chosen of
    Unlaunched reason -> ["outcome" .= ("unlaunched" :: Text), "reason" .= reason]
    Exited code read' -> ["outcome" .= ("exited" :: Text), "status" .= statusOf code, "output" .= outputName read']
    Stopped read' -> ["outcome" .= ("stopped" :: Text), "output" .= outputName read']
  where
    statusOf ExitSuccess = 0 :: Int
    statusOf (ExitFailure status) = status

outcome :: Object -> Parser Outcome
outcome fields = do
    kind <- fields .: "outcome"
    case kind :: Text of
        "unlaunched" -> Json.fields ["entry", "process", "outcome", "reason"] fields >> Unlaunched <$> fields .: "reason"
        "exited" -> Json.fields ["entry", "process", "outcome", "status", "output"] fields >> Exited . (\status -> if status == 0 then ExitSuccess else ExitFailure status) <$> fields .: "status" <*> (fields .: "output" >>= outputOf)
        "stopped" -> Json.fields ["entry", "process", "outcome", "output"] fields >> Stopped <$> (fields .: "output" >>= outputOf)
        _ -> fail ("Unknown process outcome: " ++ show kind)

outputName :: Output -> Text
outputName Complete = "complete"
outputName Cut = "cut"

outputOf :: Text -> Parser Output
outputOf "complete" = pure Complete
outputOf "cut" = pure Cut
outputOf other = fail ("Unknown process output: " ++ show other)

claimValue :: Claim -> Value
claimValue event = case event of
    Connected worker used -> object ["kind" .= ("connected" :: Text), "worker" .= worker, "epoch" .= used]
    Lost worker used -> object ["kind" .= ("lost" :: Text), "worker" .= worker, "epoch" .= used]
    Started worker used request -> object ["kind" .= ("started" :: Text), "worker" .= worker, "epoch" .= used, "request" .= request]
    Completed worker used request digest -> object ["kind" .= ("completed" :: Text), "worker" .= worker, "epoch" .= used, "request" .= request, "digest" .= digest]
    Ready update attempt bound identity before -> addressed "ready" update attempt ["binding" .= Wire.bindingValue bound, "plan" .= identity, "before" .= before]
    Current update attempt index before -> addressed "current" update attempt ["step" .= index, "before" .= before]
    Applied update attempt bound identity index consumed before after -> addressed "applied" update attempt ["binding" .= Wire.bindingValue bound, "plan" .= identity, "step" .= index, "consumed" .= consumed, "before" .= before, "after" .= after]
    Staged update attempt digest -> addressed "staged" update attempt ["digest" .= digest]
    Recorded update attempt -> addressed "recorded" update attempt []
    Committed update attempt -> addressed "committed" update attempt []
    Abandoned update attempt -> addressed "abandoned" update attempt []

claimed :: Object -> Parser Claim
claimed fields = do
    kind <- fields .: "kind"
    let update = fields .: "update"
        attempt = fields .: "attempt"
        exactly keys = Json.fields ("kind" : keys) fields
        addressedBy keys = exactly (["update", "attempt"] ++ keys)
    case kind :: Text of
        "connected" -> exactly ["worker", "epoch"] >> Connected <$> fields .: "worker" <*> fields .: "epoch"
        "lost" -> exactly ["worker", "epoch"] >> Lost <$> fields .: "worker" <*> fields .: "epoch"
        "started" -> exactly ["worker", "epoch", "request"] >> Started <$> fields .: "worker" <*> fields .: "epoch" <*> fields .: "request"
        "completed" -> exactly ["worker", "epoch", "request", "digest"] >> Completed <$> fields .: "worker" <*> fields .: "epoch" <*> fields .: "request" <*> fields .: "digest"
        "ready" -> addressedBy ["binding", "plan", "before"] >> Ready <$> update <*> attempt <*> binding fields <*> fields .: "plan" <*> fields .: "before"
        "current" -> addressedBy ["step", "before"] >> Current <$> update <*> attempt <*> fields .: "step" <*> fields .: "before"
        "applied" -> addressedBy ["binding", "plan", "step", "consumed", "before", "after"] >> Applied <$> update <*> attempt <*> binding fields <*> fields .: "plan" <*> fields .: "step" <*> fields .: "consumed" <*> fields .: "before" <*> fields .: "after"
        "staged" -> addressedBy ["digest"] >> Staged <$> update <*> attempt <*> fields .: "digest"
        "recorded" -> addressedBy [] >> Recorded <$> update <*> attempt
        "committed" -> addressedBy [] >> Committed <$> update <*> attempt
        "abandoned" -> addressedBy [] >> Abandoned <$> update <*> attempt
        _ -> fail ("Unknown journaled event: " ++ show kind)

addressed :: Text -> Natural -> Natural -> [Pair] -> Value
addressed kind update attempt rest = object (["kind" .= kind, "update" .= update, "attempt" .= attempt] ++ rest)
