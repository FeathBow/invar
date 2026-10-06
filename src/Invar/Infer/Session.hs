{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Session (Protocol (..), Declaration (..), Request (..), Input (..), Product (..), Error (..), Session, start, step, settled) where

import Control.Monad (unless, void, when)
import Data.Aeson (Object, Value (..), withObject)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.List (find, isPrefixOf)
import Data.Text (Text)
import Invar.Infer qualified as I
import Invar.Infer.Batch qualified as Batch
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Output qualified as Output
import Invar.Infer.Records qualified as Records
import Invar.Infer.Result qualified as R
import Invar.Infer.Trajectory (Trajectory)
import Invar.Infer.Trajectory.Internal qualified as Internal
import Invar.Json qualified as Json
import Invar.Measurement.Duration qualified as Duration
import Invar.Resident qualified as Boundary
import Invar.Resident.Owner qualified as Owner
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as L
import Invar.Transcript qualified as Transcript
import System.Exit (ExitCode (..))

data Protocol = Single | Serial | Batched | Resident
    deriving (Eq, Show)

data Declaration = Declaration {calls :: [Call.Call], reference :: Maybe String}

newtype Request = Request {requested :: [Call.Call]}

data Input = Dispatched Declaration | Hosted Owner.State Declaration | Line ByteString | Fragment ByteString | Ended Transcript.Outcome | Delimited

data Product = SendRequest Request | Send ByteString | Close | Admitted [Trajectory] | Owned Owner.State

data Error = Exited ExitCode | Invalid Call.Error | Protocol String
    deriving (Eq, Show)

data Session = Session {protocol :: Protocol, registry :: L.Registry, declared :: Maybe String, host :: Maybe Owner.State, phase :: Phase, completed :: [Trajectory]}

data Phase
    = Idle
    | Calling Bool [Call.Call] Call.Call [Framing.Frame]
    | Granted [Call.Call] Call.Call Call.Permit [Framing.Frame]
    | Readying [Call.Call] [Framing.Frame]
    | Executing [(Call.Call, Records.Identities, Object)] Batch.Permit [Framing.Frame] [Framing.Frame]
    | Releasing Boundary.Release (ByteString -> Owner.Released) [Trajectory]
    | Draining

start :: Protocol -> Session
start selected = Session selected L.empty Nothing Nothing Idle []

step :: Session -> Input -> Either Error (Session, [Product])
step session supplied = case supplied of
    Dispatched declaration -> dispatch session declaration
    Hosted current declaration
        | protocol session == Resident -> dispatch session {host = Just current} declaration
        | otherwise -> Left (Protocol "Only a resident session runs on a physical owner")
    Line raw -> protocolError (record raw) >>= receive session
    Fragment _ -> Left (Protocol "Worker output ends with an incomplete record")
    Ended outcome -> end session outcome
    Delimited -> case phase session of
        Draining -> pure (session {phase = Idle, completed = []}, [Admitted (completed session)])
        _ -> Left (Protocol "Session segment ends before a complete response")

dispatch :: Session -> Declaration -> Either Error (Session, [Product])
dispatch session declaration = case (phase session, protocol session, calls declaration) of
    (Idle, Single, [single]) -> serial [] single
    (Idle, Serial, selected : remaining) -> serial remaining selected
    (Idle, Batched, selected@(_ : _)) -> grouped selected
    (Idle, Resident, selected@(_ : _)) | Just _ <- host session -> grouped selected
    (Idle, _, _) -> Left (Protocol "The declared calls do not fit the session protocol")
    _ -> Left (Protocol "Calls were declared while another exchange is in flight")
  where
    serial remaining selected = pure (session {declared = reference declaration, phase = Calling True remaining selected []}, [SendRequest (Request [selected])])
    grouped selected = pure (session {declared = reference declaration, phase = Readying selected []}, [SendRequest (Request selected)])

receive :: Session -> Framing.Frame -> Either Error (Session, [Product])
receive session current = case phase session of
    Idle -> Left (Protocol "Worker output precedes its request")
    Calling initial remaining selected frames -> calling session (initial, remaining, selected, frames) current
    Granted remaining selected permit frames -> granted session (remaining, selected, permit, frames) current
    Readying selected frames -> readying session (selected, frames) current
    Executing members permit ready frames -> executing session (members, permit, ready, frames) current
    Releasing prepared released pending -> releasing session (prepared, released, pending) current
    Draining -> Left (Protocol "Output follows the final batch response")

calling :: Session -> (Bool, [Call.Call], Call.Call, [Framing.Frame]) -> Framing.Frame -> Either Error (Session, [Product])
calling session (initial, remaining, selected, frames) current = do
    let seen = map stage frames
        name = stage current
        next = frames ++ [current]
    case name of
        _
            | name `elem` ["loading", "profile", "load"] -> do
                unless (initial && any ((seen ++ [name]) `isPrefixOf`) loadings) (Left (Protocol "Missing, duplicated or reordered model-loading observations"))
                unphased current
                when (name == "load") (timed current)
                pure (session {phase = Calling initial remaining selected next}, [])
        "unloaded_adapter" -> do
            unless (not initial && null frames) (Left (Protocol "Missing or reordered adapter unload within the inference session"))
            content (Records.unloaded (Framing.fields current))
            pure (session {phase = Calling initial remaining selected next}, [])
        "loaded_adapter" -> do
            unless (if initial then seen `elem` loadings else seen == ["unloaded_adapter"]) (Left (Protocol "Missing or reordered inference observation stage"))
            expected <- expectedInput selected
            found <- content (Records.loaded (Call.plan selected, Call.binding selected, expected) (Framing.fields current))
            mapM_ (\profiled -> content (Records.profile (Framing.fields profiled) found)) (filter ((== "profile") . stage) frames)
            pure (session {phase = Calling initial remaining selected next}, [])
        "consumed" -> do
            unless (lastStage seen == Just "loaded_adapter") (Left (Protocol "Missing or reordered inference observation stage"))
            expected <- expectedInput selected
            content (Records.consumed expected (Framing.fields current))
            (updated, permit) <- invalid (Call.authorize (registry session) selected (Framing.encode next))
            pure (session {registry = updated, phase = Granted remaining selected permit next}, [Send (Call.permission permit)])
        _ -> Left (Protocol "Missing or reordered inference observation stage")

granted :: Session -> ([Call.Call], Call.Call, Call.Permit, [Framing.Frame]) -> Framing.Frame -> Either Error (Session, [Product])
granted session (remaining, selected, permit, frames) current = do
    let next = frames ++ [current]
    case (lastStage (map stage frames), stage current) of
        (Just "consumed", "inference") -> do
            unphased current
            timed current
            pure (session {phase = Granted remaining selected permit next}, [])
        (Just "inference", "result") -> do
            content (Records.result (Call.binding selected) (Framing.fields current))
            (finished, observed) <- invalid (Call.observe permit (Framing.encode next))
            content (Output.rawBehavior (Framing.raw current) (R.behaviorBits observed))
            found <- loadedIdentities selected frames
            trajectory <- admitted session (selected, found, Call.loadFact permit) (finished, observed)
            let held = session {completed = completed session ++ [trajectory]}
            case remaining of
                following : rest -> pure (held {phase = Calling False rest following []}, [SendRequest (Request [following])])
                [] -> pure (held {phase = Draining}, [Close])
        _ -> Left (Protocol "Missing or reordered inference observation stage")

readying :: Session -> ([Call.Call], [Framing.Frame]) -> Framing.Frame -> Either Error (Session, [Product])
readying session (selected, frames) current
    | Framing.grouped current = do
        let ready = frames ++ [current]
            activation = maybe False (not . Owner.initial) (host session)
        (updated, permit) <- invalid ((if activation then Batch.authorizeActivation else Batch.authorize) (registry session) selected (Framing.encode ready))
        sources <- protocolError ((if activation then Framing.activationReadiness else Framing.readiness) ready)
        members <- traverse member (zip selected sources)
        unless (protocol session == Resident) (mapM_ (\(_, found, _) -> mapM_ (\profiled -> content (Records.profile (Framing.fields profiled) found)) (filter ((== "profile") . stage) frames)) members)
        pure (session {registry = updated, phase = Executing members permit ready []}, [Send (Batch.permission permit)])
    | stage current `elem` (if maybe False (not . Owner.initial) (host session) then ["activation"] else ["loading", "profile", "load"]) = pure (session {phase = Readying selected (frames ++ [current])}, [])
    | otherwise = Left (Protocol "Missing or reordered finite batch observation stage")
  where
    member (call, source) = do
        records <- protocolError (Framing.decode source)
        case records of
            [loadedFrame, consumedFrame] -> do
                expected <- expectedInput call
                content (Records.consumed expected (Framing.fields consumedFrame))
                found <- content (Records.loaded (Call.plan call, Call.binding call, expected) (Framing.fields loadedFrame))
                pure (call, found, Framing.fields loadedFrame)
            _ -> Left (Protocol "Incomplete or repeated batch member observations")

executing :: Session -> ([(Call.Call, Records.Identities, Object)], Batch.Permit, [Framing.Frame], [Framing.Frame]) -> Framing.Frame -> Either Error (Session, [Product])
executing session (members, permit, ready, frames) current = case (map stage frames, stage current) of
    ([], "inference") -> pure (session {phase = Executing members permit ready [current]}, [])
    (["inference"], "result") | Framing.grouped current -> do
        let execution = frames ++ [current]
        observed <- invalid (Batch.observe permit (Framing.encode (ready ++ execution)))
        (_, outputs) <- protocolError (Framing.completion execution)
        unless (length outputs == length members && length observed == length members) (Left (Protocol "Batch completion inventory differs from its declared calls"))
        trajectories <- traverse admit (zip3 members outputs observed)
        case host session of
            Just physical -> do
                prepared <- protocolError (Boundary.prepare (Owner.owner physical) [fact | (_, _, fact) <- observed] (Framing.encode (ready ++ execution)))
                let released = Owner.Released (map Framing.raw (ready ++ execution)) [Call.binding call | (call, _, _) <- members] [loadedFields | (_, _, loadedFields) <- members]
                pure (session {phase = Releasing prepared released trajectories}, [Send (Boundary.request prepared)])
            Nothing -> pure (session {phase = Draining, completed = completed session ++ trajectories}, [Close])
    _ -> Left (Protocol "Missing or reordered finite batch observation stage")
  where
    admit ((call, found, _), output, (finished, result, fact)) = do
        records <- protocolError (Framing.decode output)
        case records of
            [single] -> content (Records.result (Call.binding call) (Framing.fields single))
            _ -> Left (Protocol "Incomplete or repeated batch member observations")
        admitted session (call, found, fact) (finished, result)

releasing :: Session -> (Boundary.Release, ByteString -> Owner.Released, [Trajectory]) -> Framing.Frame -> Either Error (Session, [Product])
releasing session (prepared, released, pending) current = case host session of
    Just physical -> do
        updated <- protocolError (Boundary.retire prepared (registry session) (Framing.raw current))
        following <- protocolError (Owner.release physical (released (Framing.raw current)))
        pure (session {registry = updated, host = Nothing, phase = Idle}, [Owned following, Admitted pending])
    Nothing -> Left (Protocol "A resident release has no physical owner")

settled :: Session -> Bool
settled session = case phase session of
    Idle -> null (L.active (registry session))
    _ -> False

end :: Session -> Transcript.Outcome -> Either Error (Session, [Product])
end session outcome = case (phase session, outcome) of
    (Draining, Transcript.Exited ExitSuccess Transcript.Complete) -> pure (session {phase = Idle, completed = []}, [Admitted (completed session)])
    (_, Transcript.Exited (ExitFailure status) _) -> Left (Exited (ExitFailure status))
    (Draining, _) -> Left (Protocol "Worker did not exit cleanly after its final response")
    (_, Transcript.Exited ExitSuccess _) -> Left (Protocol "Worker exited before a complete response")
    _ -> Left (Protocol "Worker ended before a complete response")

admitted :: Session -> (Call.Call, Records.Identities, L.Fact) -> (V.Completion, R.Result) -> Either Error Trajectory
admitted session (selected, found, fact) (finished, observed) = do
    case (declared session, R.referenceScores observed) of
        (Nothing, Nothing) -> pure ()
        (Just expected, Just actual) | Output.adapter actual == expected -> pure ()
        _ -> Left (Protocol "Reference scores differ from the declared reference")
    let planned = Call.plan selected
        constraint = maybe Internal.Materialization Internal.Described (I.boundPolicy planned)
        seen = Internal.Observed (Records.requested found) (Records.adapter found) (Records.tokenizer found) (Records.base found) (Records.assembly found) (Records.model found) (Records.revision found) fact
    pure (Internal.Trajectory finished constraint seen observed)

loadedIdentities :: Call.Call -> [Framing.Frame] -> Either Error Records.Identities
loadedIdentities selected frames = case find ((== "loaded_adapter") . stage) frames of
    Just loaded -> do
        expected <- expectedInput selected
        content (Records.loaded (Call.plan selected, Call.binding selected, expected) (Framing.fields loaded))
    Nothing -> Left (Protocol "Missing adapter load report")

expectedInput :: Call.Call -> Either Error Object
expectedInput selected = protocolError (Json.decode (Call.batchInput selected) >>= parseEither (withObject "expected inference consumption" pure))

record :: ByteString -> Either String Framing.Frame
record raw = Framing.Frame raw <$> (Json.decode raw >>= parseEither (withObject "worker observation" pure))

stage :: Framing.Frame -> Text
stage current = case Fields.lookup "stage" (Framing.fields current) of
    Just (String name) -> name
    _ -> ""

lastStage :: [Text] -> Maybe Text
lastStage [] = Nothing
lastStage values = Just (last values)

loadings :: [[Text]]
loadings = [["load"], ["loading", "profile", "load"]]

unphased :: Framing.Frame -> Either Error ()
unphased current = when (Fields.member "phase" (Framing.fields current)) (Left (Protocol "Unexpected phase in worker observations"))

timed :: Framing.Frame -> Either Error ()
timed current = void (content (Duration.admit (Framing.raw current) (Framing.fields current)))

protocolError :: Either String value -> Either Error value
protocolError = first Protocol

invalid :: Either Call.Error value -> Either Error value
invalid = first Invalid

content :: Either String value -> Either Error value
content = first (Invalid . Call.Protocol)
