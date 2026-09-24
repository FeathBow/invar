{-# LANGUAGE OverloadedStrings #-}

module Invar.History.Execution (Mode (..), State, Frame (..), start, validate, finish, encoded, modelLoads) where

import Control.Monad (foldM, unless, void, when)
import Data.Aeson (Object, Value (..), (.:))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.Maybe (catMaybes)
import Invar.History.Cohort qualified as Cohort
import Invar.History.Profile qualified as Profile
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Learn qualified as Learn
import Invar.Learn.Trace qualified as Learner
import Invar.Measurement.Duration qualified as Duration
import Invar.Resident qualified as Boundary
import Invar.Resident.Observation qualified as Resident
import Invar.Spec.Invocation qualified as V
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)

data Mode = Finite | Resident | Shared deriving (Eq, Show)
data State = State [Maybe Resident.State] (Maybe Resident.State) | Joint Resident.State
data Frame = Frame ByteString Object

inferenceRecordCount :: Int
inferenceRecordCount = 4

start :: (Mode, Mode) -> Natural -> Either String State
start (Shared, Shared) 1 = Right (Joint (Resident.empty (Boundary.Owner Boundary.Shared 0)))
start modes _ | Shared `elem` [fst modes, snd modes] = Left "Shared history requires both roles and exactly one physical session"
start (inference, learning) sessions = State <$> traverse (select inference . Boundary.Owner Boundary.Inference) [0 .. sessions - 1] <*> select learning (Boundary.Owner Boundary.Learning 0)
  where
    select Finite _ = Right Nothing
    select Resident owner = Right (Just (Resident.empty owner))
    select Shared _ = Left "Shared history requires one joint numerical owner"

validate :: (Learn.Settings, State) -> (Workload.Cycle, Natural, Cohort.Checked) -> [Frame] -> Either String (State, [Resident.Group], [Profile.Observation])
validate (current, Joint owner) (workload, offset, observed) frames = do
    (afterInference, inference, remaining) <- Resident.inference (offset, owner) (map encoded frames)
    (batch, _) <- Framing.takeGroup (Resident.body inference)
    let expected = map (Wire.bindingValue . binding . (offset +)) (Workload.order workload)
        actual = map (Fields.lookup "binding" . Framing.fields . Framing.consumed) (Framing.members batch)
    unless (actual == map Just expected) (Left "Shared inference partition differs from the declared execution order")
    (next, learning, rest) <- Resident.learning afterInference (current, Cohort.update observed) remaining
    unless (null rest) (Left "Output follows the shared learner release before publication")
    let groups = [inference, learning]
        profiles = concatMap (Profile.fromPrefix Boundary.Shared . map Framing.fields . Resident.prefix) groups
    pure (Joint next, groups, profiles)
validate (current, State owners learner) (workload, offset, observed) frames = do
    let count = length owners
        partitions = [[index | (position, index) <- zip [0 ..] (Workload.order workload), position `mod` count == slot] | slot <- [0 .. count - 1]]
    (inferences, groups, profiles, rest) <- foldM session ([], [], [], frames) (zip owners partitions)
    (next, learningGroups, learningProfiles) <- learn learner rest
    pure (State (reverse inferences) next, reverse groups ++ learningGroups, reverse profiles ++ learningProfiles)
  where
    session (accepted, groups, profiles, remaining) (owner, selected) = case owner of
        Nothing -> do
            when (null selected) (Left "Declared finite training session has no cohort member")
            (prefix, rest) <- inferenceSession offset remaining selected
            pure (Nothing : accepted, groups, Profile.fromPrefix Boundary.Inference prefix ++ profiles, rest)
        Just state | null selected -> pure (Just state : accepted, groups, profiles, remaining)
        Just state -> do
            (next, group, rest) <- Resident.inference (offset, state) (map encoded remaining)
            (batch, _) <- Framing.takeGroup (Resident.body group)
            let expected = map (Wire.bindingValue . binding . (offset +)) selected
                actual = map (Fields.lookup "binding" . Framing.fields . Framing.consumed) (Framing.members batch)
            unless (actual == map Just expected) (Left "Resident partition differs from the declared physical owner")
            pure (Just next : accepted, group : groups, observedProfiles Boundary.Inference group ++ profiles, map decoded rest)
    learn Nothing remaining = do
        prefix <- finiteLearning current observed remaining
        pure (Nothing, [], Profile.fromPrefix Boundary.Learning prefix)
    learn (Just state) remaining = do
        (next, group, rest) <- Resident.learning state (current, Cohort.update observed) (map encoded remaining)
        unless (null rest) (Left "Output follows the resident learner release before publication")
        pure (Just next, [group], observedProfiles Boundary.Learning group)
    observedProfiles role = Profile.fromPrefix role . map Framing.fields . Resident.prefix

finiteLearning :: Learn.Settings -> Cohort.Checked -> [Frame] -> Either String [Object]
finiteLearning current observed frames = do
    (prefix, learning) <- loadingPrefix frames
    case learning of
        Frame _ loaded : _ -> do
            profileCorrespondence prefix loaded
            Learner.validate current (Cohort.update observed) [fields | Frame _ fields <- learning]
            mapM_ (\(Frame raw fields) -> timing raw fields) [record | record@(Frame _ fields) <- learning, Fields.lookup "stage" fields `elem` map (Just . String) ["probability_roles", "reward_update", "artifacts", "checkpoint"]]
            pure prefix
        [] -> Left "Missing learner execution after the inference sessions"

finish :: State -> [Frame] -> Either String [Frame]
finish (Joint owner) records = do
    (closed, _, remaining) <- Resident.finish owner (map encoded records)
    unless (null remaining) (Left "Output follows the final shared process close")
    pure [decoded closed]
finish (State owners learner) records = do
    let selected = catMaybes (learner : reverse owners)
    (closed, remaining) <- foldM close ([], map encoded records) selected
    unless (null remaining) (Left "Output follows the final declared training cycle or process close")
    pure (reverse closed)
  where
    close (accepted, remaining) state = do
        (record, _, rest) <- Resident.finish state remaining
        pure (decoded record : accepted, rest)

encoded :: Frame -> Framing.Frame
encoded (Frame raw fields) = Framing.Frame raw fields

decoded :: Framing.Frame -> Frame
decoded (Framing.Frame raw fields) = Frame raw fields

modelLoads :: [Frame] -> Natural
modelLoads records = fromIntegral (length [() | Frame _ fields <- records, Fields.lookup "stage" fields == Just (String "load")])

inferenceSession :: Natural -> [Frame] -> [Natural] -> Either String ([Object], [Frame])
inferenceSession offset frames selected = do
    (prefix, execution) <- loadingPrefix frames
    case execution of
        Frame initialBytes loaded : _ | Framing.grouped (Framing.Frame initialBytes loaded) -> do
            (group, rest) <- Framing.takeGroup [Framing.Frame raw fields | Frame raw fields <- execution]
            let expected = map (Wire.bindingValue . binding . (offset +)) selected
                actual = map (Fields.lookup "binding" . Framing.fields . Framing.consumed) (Framing.members group)
            unless (actual == map Just expected) (Left "Batch execution partition differs from the declared session order")
            mapM_ (profileCorrespondence prefix . Framing.fields . Framing.loaded) (Framing.members group)
            pure (prefix, [Frame (Framing.raw record) (Framing.fields record) | record <- rest])
        Frame _ loaded : _ -> do
            profileCorrespondence prefix loaded
            (_, remaining) <- foldM call (Nothing, execution) selected
            pure (prefix, remaining)
        [] -> Left "Missing inference session after its model load"
  where
    call (previous, events) index = do
        remaining <- case previous of
            Nothing -> Right events
            Just loading -> case events of
                Frame _ unloaded : rest -> do
                    unless (Fields.lookup "stage" unloaded == Just (String "unloaded_adapter")) (Left "Missing adapter unload within inference session")
                    parseEither (Json.fields ["stage", "binding", "program"]) unloaded
                    unless (Object (Fields.delete "stage" unloaded) == loading) (Left "Unloaded adapter differs from the preceding invocation")
                    pure rest
                [] -> Left "Missing adapter unload within inference session"
        let (segment, rest) = splitAt inferenceRecordCount remaining
            expected = binding (offset + index)
        unless (length segment == inferenceRecordCount) (Left "Incomplete inference session member")
        loading <- case segment of
            [Frame _ loaded, _, Frame measuredRaw measured, _] -> do
                unless (Fields.lookup "binding" loaded == Just (Wire.bindingValue expected)) (Left "Inference execution partition differs from the declared session order")
                timing measuredRaw measured
                parseEither (.: "load") loaded
            _ -> Left "Incomplete inference session member"
        pure (Just loading, rest)

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

binding :: Natural -> V.Binding
binding = V.ordinal
