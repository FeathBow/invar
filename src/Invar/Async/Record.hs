{-# LANGUAGE OverloadedStrings #-}

module Invar.Async.Record (Entry (..), Claim (..), Role (..), claim, encode, decode) where

import Data.Aeson (Object, Value (..), object, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Pair, Parser)
import Data.Text (Text)
import Data.Text qualified as Text
import Invar.Async.Completion qualified as Completion
import Invar.Async.Core (Attempt (..), Epoch (..), Worker (..))
import Invar.Async.Core qualified as Core
import Invar.Async.Plan (Request (..), Update (..), Version (..))
import Invar.Infer.Wire qualified as Wire
import Invar.Spec.Invocation qualified as V
import Invar.Transcript (Outcome (..))
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

data Entry
    = Declared FilePath Object
    | Opened Natural Natural
    | Dispatched Natural Natural Natural V.Binding
    | Attempted Natural Natural V.Binding
    | Reserved Natural Role Natural Natural
    | Finished Natural Outcome
    | Happened Claim [Core.Command]
    | Stored Natural V.Binding String
    | Verified Natural Natural String
    | Resumed [Natural] Natural Natural Natural
    | Elapsed Text Natural Double Double
    deriving (Eq, Show)

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
    Declared directory fields -> Object (Fields.insert "entry" "declaration" (Fields.insert "directory" (String (Text.pack directory)) fields))
    Opened worker used -> entry "epoch" ["worker" .= worker, "epoch" .= used]
    Dispatched request worker used bound -> entry "dispatched" ["request" .= request, "worker" .= worker, "epoch" .= used, "binding" .= Wire.bindingValue bound]
    Attempted update attempt bound -> entry "attempt" ["update" .= update, "attempt" .= attempt, "binding" .= Wire.bindingValue bound]
    Reserved number chosen slot used -> entry "process" ["process" .= number, "role" .= roleName chosen, "slot" .= slot, "epoch" .= used]
    Finished number ended -> entry "exit" (("process" .= number) : outcomeFields ended)
    Happened event commands -> entry "event" ["event" .= claimValue event, "commands" .= map commandValue commands]
    Stored request bound digest -> entry "result" ["request" .= request, "binding" .= Wire.bindingValue bound, "digest" .= digest]
    Verified update attempt learner -> entry "checkpoint" ["update" .= update, "attempt" .= attempt, "learner" .= learner]
    Resumed done used identities processes -> entry "resume" ["committed" .= done, "epoch" .= used, "identities" .= identities, "processes" .= processes]
    Elapsed kind update started ended -> entry "interval" ["role" .= kind, "update" .= update, "start" .= started, "end" .= ended]
  where
    entry name fields = object (("entry" .= (name :: Text)) : fields)

decode :: Object -> Parser Entry
decode fields = do
    kind <- fields .: "entry"
    case kind :: Text of
        "declaration" -> Declared <$> fields .: "directory" <*> pure (Fields.delete "entry" (Fields.delete "directory" fields))
        "epoch" -> Opened <$> fields .: "worker" <*> fields .: "epoch"
        "dispatched" -> Dispatched <$> fields .: "request" <*> fields .: "worker" <*> fields .: "epoch" <*> Wire.binding fields
        "attempt" -> Attempted <$> fields .: "update" <*> fields .: "attempt" <*> Wire.binding fields
        "process" -> Reserved <$> fields .: "process" <*> (fields .: "role" >>= roleOf) <*> fields .: "slot" <*> fields .: "epoch"
        "exit" -> Finished <$> fields .: "process" <*> outcome fields
        "event" -> Happened <$> (fields .: "event" >>= withObject "journaled event" claimed) <*> (fields .: "commands" >>= traverse (withObject "journaled command" command))
        "result" -> Stored <$> fields .: "request" <*> Wire.binding fields <*> fields .: "digest"
        "checkpoint" -> Verified <$> fields .: "update" <*> fields .: "attempt" <*> fields .: "learner"
        "resume" -> Resumed <$> fields .: "committed" <*> fields .: "epoch" <*> fields .: "identities" <*> fields .: "processes"
        "interval" -> Elapsed <$> fields .: "role" <*> fields .: "update" <*> fields .: "start" <*> fields .: "end"
        _ -> fail ("Unknown journal entry: " ++ show kind)

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
    Exited ExitSuccess -> ["outcome" .= ("exited" :: Text), "status" .= (0 :: Int)]
    Exited (ExitFailure status) -> ["outcome" .= ("exited" :: Text), "status" .= status]
    Stopped -> ["outcome" .= ("stopped" :: Text)]

outcome :: Object -> Parser Outcome
outcome fields = do
    kind <- fields .: "outcome"
    case kind :: Text of
        "unlaunched" -> Unlaunched <$> fields .: "reason"
        "exited" -> Exited . (\status -> if status == 0 then ExitSuccess else ExitFailure status) <$> fields .: "status"
        "stopped" -> pure Stopped
        _ -> fail ("Unknown process outcome: " ++ show kind)

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
    case kind :: Text of
        "connected" -> Connected <$> fields .: "worker" <*> fields .: "epoch"
        "lost" -> Lost <$> fields .: "worker" <*> fields .: "epoch"
        "started" -> Started <$> fields .: "worker" <*> fields .: "epoch" <*> fields .: "request"
        "completed" -> Completed <$> fields .: "worker" <*> fields .: "epoch" <*> fields .: "request" <*> fields .: "digest"
        "ready" -> Ready <$> update <*> attempt <*> Wire.binding fields <*> fields .: "plan" <*> fields .: "before"
        "current" -> Current <$> update <*> attempt <*> fields .: "step" <*> fields .: "before"
        "applied" -> Applied <$> update <*> attempt <*> Wire.binding fields <*> fields .: "plan" <*> fields .: "step" <*> fields .: "consumed" <*> fields .: "before" <*> fields .: "after"
        "staged" -> Staged <$> update <*> attempt <*> fields .: "digest"
        "recorded" -> Recorded <$> update <*> attempt
        "committed" -> Committed <$> update <*> attempt
        "abandoned" -> Abandoned <$> update <*> attempt
        _ -> fail ("Unknown journaled event: " ++ show kind)

commandValue :: Core.Command -> Value
commandValue issued = case issued of
    Core.Dispatch (Request request) (Version version) -> object ["kind" .= ("dispatch" :: Text), "request" .= request, "version" .= version]
    Core.Send (Update update) (Attempt attempt) -> addressed "send" update attempt []
    Core.Open (Update update) (Attempt attempt) index before -> addressed "open" update attempt ["step" .= index, "before" .= before]
    Core.Record (Update update) (Attempt attempt) digest -> addressed "record" update attempt ["digest" .= digest]
    Core.Commit (Update update) (Attempt attempt) -> addressed "commit" update attempt []

command :: Object -> Parser Core.Command
command fields = do
    kind <- fields .: "kind"
    let update = Update <$> fields .: "update"
        attempt = Attempt <$> fields .: "attempt"
    case kind :: Text of
        "dispatch" -> Core.Dispatch . Request <$> fields .: "request" <*> (Version <$> fields .: "version")
        "send" -> Core.Send <$> update <*> attempt
        "open" -> Core.Open <$> update <*> attempt <*> fields .: "step" <*> fields .: "before"
        "record" -> Core.Record <$> update <*> attempt <*> fields .: "digest"
        "commit" -> Core.Commit <$> update <*> attempt
        _ -> fail ("Unknown journaled command: " ++ show kind)

addressed :: Text -> Natural -> Natural -> [Pair] -> Value
addressed kind update attempt rest = object (["kind" .= kind, "update" .= update, "attempt" .= attempt] ++ rest)
