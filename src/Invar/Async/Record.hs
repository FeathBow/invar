{-# LANGUAGE OverloadedStrings #-}

module Invar.Async.Record (Entry (..), Claim (..), Role (..), claim, encode, decode) where

import Data.Aeson (Object, Value (..), object, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Pair, Parser)
import Data.Text (Text)
import Data.Text qualified as Text
import Invar.Async.Core (Attempt (..))
import Invar.Async.Core qualified as Core
import Invar.Async.Entry (Claim (..), Role (..), claim, claimValue, claimed, outcome, outcomeFields, roleName, roleOf)
import Invar.Async.Plan (Request (..), Update (..), Version (..))
import Invar.Infer.Wire qualified as Wire
import Invar.Spec.Invocation qualified as V
import Invar.Transcript (Outcome (..))
import Numeric.Natural (Natural)

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
