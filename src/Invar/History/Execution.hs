{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Execution (Mode (..), State, Frame (..), Declared (..), Cycle (..), start, validate, finish, encoded, modelLoads) where

import Control.Monad (foldM, unless, void, when)
import Data.Aeson (Object, Value (..), (.:))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.Maybe (catMaybes)
import Invar.History.Profile qualified as Profile
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Session qualified as Session
import Invar.Infer.Wire qualified as Wire
import Invar.Learn qualified as Learn
import Invar.Learn.Framing qualified as Learner
import Invar.Learn.Report qualified as Report
import Invar.Learn.Trace qualified as Trace
import Invar.Measurement.Duration qualified as Duration
import Invar.Resident qualified as Boundary
import Invar.Resident.Group qualified as Group
import Invar.Resident.Owner qualified as Owner
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)

data Mode = Finite | Batched | Resident | Shared deriving (Eq, Show)
data State = State Session.Protocol [Maybe Hosted] (Maybe Owner.State) | Joint Hosted
data Frame = Frame ByteString Object
data Declared = Declared {calls :: [Call.Call], reference :: Maybe String, report :: Report.Report}
data Cycle = Cycle {pending :: [Replay.Pending], admitted :: [Replay.Logged], groups :: [Group.Group], profiles :: [Profile.Observation]}

type Hosted = (Session.Session, Owner.State)

start :: (Mode, Mode) -> Natural -> Either String State
start (Shared, Shared) 1 = Right (Joint (hosted (Boundary.Owner Boundary.Shared 0)))
start modes _ | Shared `elem` [fst modes, snd modes] = Left "Shared history requires both roles and exactly one physical session"
start (inference, learning) sessions = do
    resident <- case learning of
        Finite -> Right Nothing
        Resident -> Right (Just (Owner.start (Boundary.Owner Boundary.Learning 0)))
        _ -> Left "A learner runs as a process, a resident owner or a shared owner"
    let protocol = if inference == Batched then Session.Batched else Session.Serial
    pure (State protocol [if inference == Resident then Just (hosted (Boundary.Owner Boundary.Inference index)) else Nothing | index <- [0 .. sessions - 1]] resident)

hosted :: Boundary.Owner -> Hosted
hosted selected = (Session.start Session.Resident, Owner.start selected)

validate :: (Learn.Settings, State) -> (Workload.Cycle, Declared) -> [Frame] -> Either String (State, Cycle)
validate (current, Joint owner) (workload, declared) frames = do
    (afterInference, logged, inference, remaining) <- group owner (ordered (Workload.order workload) declared) (map encoded frames)
    (physical, learning, rest) <- learner (snd afterInference) (current, report declared) remaining
    unless (null rest) (Left "Output follows the shared learner release before publication")
    let observed = [inference, learning]
        modelProfiles = concatMap (Profile.fromPrefix Boundary.Shared . map Framing.fields . Group.prefix) observed
    pure (Joint (fst afterInference, physical), Cycle [] logged observed modelProfiles)
validate (current, State protocol owners learning) (workload, declared) frames = do
    let count = length owners
        partitions = [[index | (position, index) <- zip [0 ..] (Workload.order workload), position `mod` count == slot] | slot <- [0 .. count - 1]]
    (sessions, collected, rest) <- foldM session ([], Cycle [] [] [] [], map encoded frames) (zip owners partitions)
    (next, learned) <- learn learning rest
    pure (State protocol (reverse sessions) next, learned collected)
  where
    session (accepted, collected, remaining) (owner, selected) = case owner of
        Nothing -> do
            when (null selected) (Left "Declared finite training session has no cohort member")
            (waiting, consumed, rest) <- first show (Replay.session protocol (ordered selected declared) remaining)
            pure (Nothing : accepted, collected {pending = pending collected ++ [waiting], profiles = profiles collected ++ Profile.fromPrefix Boundary.Inference (preparation consumed)}, rest)
        Just state | null selected -> pure (Just state : accepted, collected, remaining)
        Just state -> do
            (next, logged, observed, rest) <- group state (ordered selected declared) remaining
            pure (Just next : accepted, collected {admitted = admitted collected ++ logged, groups = groups collected ++ [observed], profiles = profiles collected ++ Profile.fromPrefix Boundary.Inference (map Framing.fields (Group.prefix observed))}, rest)
    learn Nothing remaining = do
        prefix <- finiteLearning current (report declared) (map decoded remaining)
        pure (Nothing, \collected -> collected {profiles = profiles collected ++ Profile.fromPrefix Boundary.Learning prefix})
    learn (Just physical) remaining = do
        (next, observed, rest) <- learner physical (current, report declared) remaining
        unless (null rest) (Left "Output follows the resident learner release before publication")
        pure (Just next, \collected -> collected {groups = groups collected ++ [observed], profiles = profiles collected ++ Profile.fromPrefix Boundary.Learning (map Framing.fields (Group.prefix observed))})

ordered :: [Natural] -> Declared -> Session.Declaration
ordered selected declared = Session.Declaration [chosen | index <- selected, chosen <- take 1 (drop (fromIntegral index) (calls declared))] (reference declared)

group :: Hosted -> Session.Declaration -> [Framing.Frame] -> Either String (Hosted, [Replay.Logged], Group.Group, [Framing.Frame])
group (current, physical) declaration@(Session.Declaration chosen _) frames = do
    (next, logged, consumed, rest) <- first show (Replay.group (current, physical) declaration frames)
    (records, acknowledged) <- case reverse consumed of
        final : reversed -> pure (reverse reversed, final)
        [] -> Left "Resident group has no release acknowledgement"
    observed <- Group.observe (Owner.owner physical, Owner.groups physical, map Call.binding chosen) records acknowledged
    pure (next, logged, observed, rest)

learner :: Owner.State -> (Learn.Settings, Report.Report) -> [Framing.Frame] -> Either String (Owner.State, Group.Group, [Framing.Frame])
learner physical (settings, reported) records = do
    let Boundary.Owner role _ = Owner.owner physical
        initial = Owner.initial physical
        (leading, remaining) = span ((`elem` map (Just . String) ["loading", "profile", "load", "activation"]) . Framing.stageName) records
    unless (role `elem` [Boundary.Learning, Boundary.Shared]) (Left "Resident observation has a different numerical owner role")
    (execution, rest) <- case break ((== Just (String "result")) . Framing.stageName) remaining of
        (preceding, result : following) -> pure (preceding ++ [result], following)
        _ -> Left "Incomplete resident update result"
    let (ready, _) = break ((== Just (String "consumed")) . Framing.stageName) execution
    consumed <- case drop (length ready) execution of
        value : _ -> pure value
        [] -> Left "Missing resident learner consumption"
    loaded <- case execution of
        value : _ -> pure value
        [] -> Left "Missing resident learner execution"
    Learner.readiness initial (Report.request reported) (Framing.encode (leading ++ ready ++ [consumed]))
    Learner.completion (Report.checkedRequest reported) (Framing.encode (leading ++ execution))
    Trace.validate settings reported (map Framing.fields execution)
    (acknowledged, after) <- case rest of
        value : following -> pure (value, following)
        [] -> Left "Resident group has no release acknowledgement"
    bound <- parseEither Wire.binding (Framing.fields consumed)
    loads <- parseEither (.: "load") (Framing.fields consumed)
    void (Boundary.observeRelease (Owner.owner physical, [loads], Framing.encode (leading ++ execution)) (Framing.raw acknowledged))
    following <- Owner.release physical (Owner.Released (map Framing.raw (leading ++ execution)) [bound] [Framing.fields loaded] (Framing.raw acknowledged))
    observed <- Group.observe (Owner.owner physical, Owner.groups physical, [bound]) (leading ++ execution) acknowledged
    pure (following, observed, after)

finiteLearning :: Learn.Settings -> Report.Report -> [Frame] -> Either String [Object]
finiteLearning current reported frames = do
    (prefix, learning) <- loadingPrefix frames
    case learning of
        Frame _ loaded : _ -> do
            profileCorrespondence prefix loaded
            Trace.validate current reported [fields | Frame _ fields <- learning]
            mapM_ (\(Frame raw fields) -> timing raw fields) [record | record@(Frame _ fields) <- learning, Fields.lookup "stage" fields `elem` map (Just . String) ["reward_update", "artifacts", "checkpoint"]]
            pure prefix
        [] -> Left "Missing learner execution after the inference sessions"

finish :: State -> [Frame] -> Either String [Frame]
finish (Joint (current, physical)) records = do
    (closed, remaining) <- close ([], map encoded records) (Just current, physical)
    unless (null remaining) (Left "Output follows the final shared process close")
    pure (map decoded closed)
finish (State _ owners learning) records = do
    let selected = catMaybes (fmap (Nothing,) learning : map (fmap (first Just)) (reverse owners))
    (closed, remaining) <- foldM close ([], map encoded records) selected
    unless (null remaining) (Left "Output follows the final declared training cycle or process close")
    pure (map decoded (reverse closed))

close :: ([Framing.Frame], [Framing.Frame]) -> (Maybe Session.Session, Owner.State) -> Either String ([Framing.Frame], [Framing.Frame])
close (accepted, remaining) (current, physical) = case remaining of
    record : rest -> do
        unless (all Session.settled current) (Left "Resident process closes with active invocation loads")
        Owner.close physical (Framing.raw record)
        pure (record : accepted, rest)
    [] -> Left "Missing final resident process close"

preparation :: [Framing.Frame] -> [Object]
preparation = map Framing.fields . takeWhile ((`elem` map (Just . String) ["loading", "profile", "load"]) . Framing.stageName)

encoded :: Frame -> Framing.Frame
encoded (Frame raw fields) = Framing.Frame raw fields

decoded :: Framing.Frame -> Frame
decoded (Framing.Frame raw fields) = Frame raw fields

modelLoads :: [Frame] -> Natural
modelLoads records = fromIntegral (length [() | Frame _ fields <- records, Fields.lookup "stage" fields == Just (String "load")])

loadingPrefix :: [Frame] -> Either String ([Object], [Frame])
loadingPrefix frames = do
    let (prefix, remaining) = span (\(Frame _ fields) -> Fields.lookup "stage" fields `elem` map (Just . String) ["loading", "profile", "load"]) frames
        values = [fields | Frame _ fields <- prefix]
        stages = map (Fields.lookup "stage") values
    unless (stages `elem` map (map (Just . String)) [["load"], ["loading", "profile", "load"]] && not (any (Fields.member "phase") values)) (Left "Missing, duplicated or reordered model-loading observations")
    mapM_ (\(Frame raw fields) -> timing raw fields) [record | record@(Frame _ fields) <- prefix, Fields.lookup "stage" fields == Just (String "load")]
    pure (values, remaining)

profileCorrespondence :: [Object] -> Object -> Either String ()
profileCorrespondence prefix loaded = mapM_ match [fields | fields <- prefix, Fields.lookup "stage" fields == Just (String "profile")]
  where
    match fields = unless (all (\key -> Fields.lookup key fields == Fields.lookup key loaded) ["model", "revision"]) (Left "Model profile differs from the loaded model or revision")

timing :: ByteString -> Object -> Either String ()
timing raw = void . Duration.admit raw
