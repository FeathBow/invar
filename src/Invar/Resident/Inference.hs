{-# LANGUAGE OverloadedStrings #-}

module Invar.Resident.Inference (Ledger (..), admit, layout) where

import Control.Monad (foldM, unless, void)
import Data.Aeson (Value (..))
import Data.Aeson.KeyMap qualified as Fields
import Data.Bifunctor (first)
import Invar.Infer qualified as Infer
import Invar.Infer.Framing qualified as Frame
import Invar.Infer.Model qualified as Model
import Invar.Infer.Observation qualified as Observation
import Invar.Infer.Wire qualified as Wire
import Invar.Measurement.Duration qualified as Duration
import Invar.Resident qualified as Boundary
import Invar.Resident.Observation qualified as Resident
import Invar.Spec.Invocation qualified as Invocation
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)

data Ledger = Ledger
    { sessions :: Natural
    , groups :: [(Natural, [Resident.Group])]
    , closes :: [(Boundary.Owner, Frame.Frame, Duration.Duration)]
    }
    deriving (Eq, Show)

admit :: (Workload.Document, String, Model.Model) -> Natural -> [Frame.Frame] -> Either String Ledger
admit (tasks, policy, model) count records = do
    unless (count > 0) (Left "Resident evaluation requires physical owners")
    let initial = [Resident.empty (Boundary.Owner Boundary.Inference index) | index <- [0 .. count - 1]]
    (_, owners, completed, remaining) <- foldM advance (0, initial, [], records) (zip [0 ..] (Workload.cycles tasks))
    (closed, rest) <- foldM close ([], remaining) (reverse owners)
    unless (null rest) (Left "Output follows the final physical inference close")
    pure (Ledger count (reverse completed) (reverse closed))
  where
    advance (offset, owners, accepted, remaining) (index, workload) = do
        (next, observed, following) <- foldM (group (index, offset, workload)) ([], [], remaining) (zip [0 ..] owners)
        rest <- case following of
            record : rest | Fields.lookup "phase" (Frame.fields record) == Just (String "evaluation") -> pure rest
            _ -> Left "Resident groups do not end at their declared evaluation cohort"
        pure (offset + fromIntegral (length (Workload.tasks workload)), reverse next, (index, reverse observed) : accepted, rest)
    group (cohort, offset, workload) (owners, accepted, remaining) (slot, owner) = do
        let indices = [index | (position, index) <- zip [0 ..] (Workload.order workload), position `mod` count == slot]
        if null indices
            then pure (owner : owners, accepted, remaining)
            else do
                (next, observed, rest) <- Resident.inference (cohort, owner) remaining
                checkGroup (policy, model) (offset, workload, indices) observed
                pure (next : owners, observed : accepted, rest)
    close (accepted, remaining) owner = do
        (record, elapsed, rest) <- Resident.finish owner remaining
        pure ((Resident.owner owner, record, elapsed) : accepted, rest)

checkGroup :: (String, Model.Model) -> (Natural, Workload.Cycle, [Natural]) -> Resident.Group -> Either String ()
checkGroup (policy, model) (offset, workload, selected) observed = do
    (batch, remaining) <- Frame.takeGroup (Resident.body observed)
    unless (null remaining) (Left "Extra resident numerical group output")
    let expected = map (binding . (offset +)) selected
        actual = map (Fields.lookup "binding" . Frame.fields . Frame.consumed) (Frame.members batch)
    unless (actual == map (Just . Wire.bindingValue) expected) (Left "Resident inference group differs from its declared physical owner partition")
    mapM_ (check batch) (zip selected expected)
  where
    check batch (index, bound) = do
        task <- case drop (fromIntegral index) (Workload.tasks workload) of
            value : _ -> pure value
            [] -> Left "Resident group has no declared task"
        requested <- case model of
            Model.Materialized tokenizer base assembly -> pure (Infer.Request policy tokenizer base assembly (Workload.prompt task) (Workload.tokens task) (Workload.temperature task) (Workload.seed task))
            _ -> Left "Resident inference requires complete model materialization"
        planned <- first show (Infer.prepare requested)
        void (Observation.admitGroup planned bound batch)

binding :: Natural -> Invocation.Binding
binding index = Invocation.Binding (Invocation.CallId index) (Invocation.AttemptId index) (Invocation.Instance index)

layout :: Ledger -> [[[Int]]]
layout ledger = [[[length (Resident.bindings group) | group <- observed, Resident.physicalOwner group == Boundary.Owner Boundary.Inference slot] | slot <- [0 .. sessions ledger - 1]] | (_, observed) <- groups ledger]
