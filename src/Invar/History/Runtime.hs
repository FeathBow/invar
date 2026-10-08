{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Runtime (Checked, inspect, config, workload, generations, finalPolicy, identities, profiles, loads, describe, recorded) where

import Control.Monad (unless)
import Data.Aeson (Object, Value (..), object, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.List (genericLength)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Invar.Async.Core qualified as Core
import Invar.Async.Entry qualified as Entry
import Invar.Async.Plan (Declared (..), Request (..), Update (..), Version (..))
import Invar.Async.Plan qualified as Plan
import Invar.Async.Recorded qualified as Recorded
import Invar.Async.Replay qualified as Replay
import Invar.History.Generation qualified as Generation
import Invar.History.Profile qualified as Profile
import Invar.Infer.Wire qualified as Wire
import Invar.Journal qualified as Journal
import Invar.Loop qualified as Loop
import Invar.Policy qualified as Policy
import Invar.Resident qualified as Resident
import Invar.Runtime qualified as Runtime
import Invar.Store qualified as Store
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)
import System.Directory (canonicalizePath, makeAbsolute, withCurrentDirectory)
import System.FilePath ((</>))
import System.IO.Error (tryIOError)

data Checked = Checked Runtime.Run Replay.Declaration Replay.Replayed [Generation.Generation] Policy.Description [[Entry.Generation]]

inspect :: FilePath -> (Value -> Either String (Loop.Config, Natural)) -> IO (Either String Checked)
inspect directory interpreter = do
    target <- canonicalizePath directory
    Journal.inspect (target </> "journal.jsonl") $ \journaled -> case traverse (parseEither Entry.decode) journaled of
        Left problem -> pure (Left problem)
        Right (Entry.Declared started declared : later) -> do
            interpreted <- withCurrentDirectory started (Runtime.interpret target interpreter declared >>= traverse (located target))
            prepared <- either (pure . Left) (\selected -> fmap (selected,) <$> Runtime.declaration selected) interpreted
            case prepared of
                Left problem -> pure (Left (show problem))
                Right (selected, replaying) -> do
                    found <- tryIOError ((,) <$> Recorded.transcripts target later <*> Recorded.generations target)
                    pure $ case found of
                        Right (transcribed, observed) -> admit selected replaying later transcribed observed
                        Left problem -> Left ("A transcript or a generation cannot be read: " ++ show problem)
        Right _ -> pure (Left "The journal does not start with a run declaration")
  where
    located target (Runtime.Run chosen lag document) = do
        checkpoint <- makeAbsolute (Loop.checkpoint chosen)
        reference <- makeAbsolute (Loop.reference chosen)
        pure (Runtime.Run chosen {Loop.root = target, Loop.checkpoint = checkpoint, Loop.reference = reference} lag document)

admit :: Runtime.Run -> Replay.Declaration -> [Entry.Entry] -> Map Natural ByteString -> [Entry.Generation] -> Either String Checked
admit selected declared entries transcribed observed = do
    replayed <- Replay.replay declared entries transcribed
    let chosen = Replay.config declared
        updates = Plan.updates (Replay.plan declared)
        published = [(version, policy, learner, described) | (version, (policy, learner, described)) <- Map.toList (Replay.versions replayed), version > 0]
    unless (Core.committed (Replay.state replayed) == updates) (Left "The run has not committed every declared update")
    unless (null (Replay.unended replayed)) (Left ("A process has no recorded end: " ++ unwords (map show (Replay.unended replayed))))
    unless ([(Entry.version seen, Entry.adapter seen, Entry.learner seen, Entry.description seen) | seen <- observed] == published) (Left "The published generations differ from the committed updates")
    unless (Map.keys (Replay.evidence replayed) == [1 .. genericLength updates]) (Left "A committed update has no evidence")
    built <- traverse (generation chosen) (Map.elems (Replay.evidence replayed))
    final <- case reverse published of
        (_, _, _, described) : _ -> Right described
        [] -> Left "A runtime history requires a committed update"
    pure (Checked selected declared replayed built final [seen | Entry.Restarted seen <- entries])
  where
    generation chosen evidenced =
        let (settings, tasks) = Replay.admittedUnder evidenced
         in Generation.generation settings (Loop.root chosen, Store.methodName (Loop.publication chosen)) (tasks, Replay.trajectories evidenced) (Replay.execution evidenced) ([], [], [])

config :: Checked -> Loop.Config
config (Checked (Runtime.Run chosen _ _) _ _ _ _ _) = chosen

workload :: Checked -> Workload.Document
workload (Checked (Runtime.Run _ _ document) _ _ _ _ _) = document

generations :: Checked -> [Generation.Generation]
generations (Checked _ _ _ built _ _) = built

finalPolicy :: Checked -> Policy.Description
finalPolicy (Checked _ _ _ _ final _) = final

identities :: Checked -> Natural
identities (Checked _ _ replayed _ _ _) = Replay.identity (Replay.floors replayed)

profiles :: Checked -> [Profile.Observation]
profiles (Checked _ _ replayed _ _ _) = concat [Profile.fromPrefix (role chosen) loading | (chosen, loading) <- Map.elems (Replay.loaded replayed)]
  where
    role Entry.Inference = Resident.Inference
    role Entry.Learner = Resident.Learning

loads :: Checked -> [Value]
loads (Checked _ _ replayed _ _ _) = [Object fields | (_, loading) <- Map.elems (Replay.loaded replayed), fields <- loading]

describe :: Checked -> Value
describe (Checked _ declared replayed built _ restarts) =
    object
        [ "staleness" .= Replay.staleness declared
        , "updates" .= [object ["update" .= update, "version" .= selected, "requests" .= [request | Request request <- requests]] | Update update <- Plan.updates (Replay.plan declared), let Version selected = Plan.version (Replay.plan declared) (Update update), Just (Declared requests _) <- [Plan.declared (Replay.plan declared) (Update update)]]
        , "attempts" .= attempts replayed
        , "restarts" .= [[object ["version" .= Entry.version seen, "adapter" .= Entry.adapter seen, "learner" .= Entry.learner seen] | seen <- observed] | observed <- restarts]
        , "generations" .= map Generation.describe built
        ]

recorded :: Checked -> Value
recorded (Checked _ _ replayed _ _ restarts) = object ["source" .= ("run directory" :: Text), "processes" .= length (Replay.reserved replayed), "attempts" .= attempts replayed, "restarts" .= length restarts]

attempts :: Replay.Replayed -> [Value]
attempts replayed = [object ["update" .= update, "attempt" .= tried, "binding" .= Wire.bindingValue (Replay.binding learned), "process" .= Replay.process learned, "consumed" .= map Wire.bindingValue (Replay.consumed learned), "outcome" .= outcome (Replay.outcome learned)] | ((update, tried), learned) <- Map.toList (Replay.learned replayed)]
  where
    outcome :: Replay.Outcome -> Object
    outcome (Replay.Committed version) = Fields.fromList ["committed" .= version]
    outcome Replay.Concluded = Fields.fromList ["concluded" .= True]
    outcome (Replay.Incomplete reason) = Fields.fromList ["incomplete" .= (reason :: String)]
