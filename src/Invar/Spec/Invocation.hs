{-# LANGUAGE Safe #-}

module Invar.Spec.Invocation (
    ordinal,
    Runtime,
    CallId (..),
    AttemptId (..),
    Instance (..),
    Selection (..),
    Binding (..),
    Consumption (..),
    Phase (..),
    Error (..),
    Completion,
    Intention,
    completedBinding,
    completedProgram,
    completedCommand,
    completedInputs,
    completedEmission,
    completedOutput,
    intendedProgram,
    intendedCommand,
    intendedInputs,
    intendedEmission,
    start,
    prepare,
    issue,
    consume,
    finish,
    cancel,
    intent,
    intention,
    phase,
    completion,
) where

import Control.Monad (unless, when)
import Data.ByteString (ByteString)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Numeric.Natural (Natural)

newtype CallId = CallId Natural
    deriving (Eq, Ord, Show)

newtype AttemptId = AttemptId Natural
    deriving (Eq, Ord, Show)

newtype Instance = Instance Natural
    deriving (Eq, Ord, Show)

data Selection = Selection {selectedCall :: CallId, inputs :: E.World}

data Binding = Binding {boundCall :: CallId, boundAttempt :: AttemptId, boundInstance :: Instance}
    deriving (Eq, Ord, Show)

ordinal :: Natural -> Binding
ordinal index = Binding (CallId index) (AttemptId index) (Instance index)

data Consumption = Consumption {binding :: Binding, program :: ByteString, emission :: E.Emission}
    deriving (Eq, Show)

data Phase = Issued | Consumed | Finished | Cancelled
    deriving (Eq, Show)

data Error
    = DuplicateCall CallId
    | UnknownCall CallId
    | InvalidInputs E.Error
    | MissingCommand Natural
    | CallInUse CallId AttemptId
    | DuplicateAttempt AttemptId
    | UnknownAttempt AttemptId
    | BindingMismatch Binding Binding
    | PhaseMismatch Phase Phase
    | ProgramMismatch
    | EmissionMismatch E.Emission E.Emission
    deriving (Eq, Show)

data Completion = Completion
    { reportBinding :: Binding
    , reportProgram :: ByteString
    , reportCommand :: Natural
    , reportInputs :: E.World
    , reportEmission :: E.Emission
    , reportOutput :: ByteString
    }
    deriving (Eq, Show)

data Intention = Intention ByteString Natural E.World E.Emission
    deriving (Eq, Show)

data Call = Call E.World E.Emission (Maybe AttemptId)
data Attempt = Attempt Binding Phase (Maybe Completion)
data Runtime = Runtime A.Checked Natural (Map CallId Call) (Map AttemptId Attempt)

start :: A.Checked -> Natural -> Runtime
start checked position = Runtime checked position Map.empty Map.empty

prepare :: Selection -> Runtime -> Either Error Runtime
prepare selection (Runtime checked position calls attempts) = do
    let name = selectedCall selection
    when (Map.member name calls) (Left (DuplicateCall name))
    emissions <- either (Left . InvalidInputs) Right (A.run checked (inputs selection))
    selected <- select position emissions
    pure (Runtime checked position (Map.insert name (Call (inputs selection) selected Nothing) calls) attempts)
  where
    select _ [] = Left (MissingCommand position)
    select 0 (value : _) = Right value
    select remaining (_ : rest) = select (remaining - 1) rest

issue :: Binding -> Runtime -> Either Error Runtime
issue bound runtime@(Runtime checked position calls attempts) = do
    Call assignments selected active <- lookupCall runtime (boundCall bound)
    case active of
        Just previous -> Left (CallInUse (boundCall bound) previous)
        Nothing -> pure ()
    when (Map.member (boundAttempt bound) attempts) (Left (DuplicateAttempt (boundAttempt bound)))
    let updated = Map.insert (boundCall bound) (Call assignments selected (Just (boundAttempt bound))) calls
    pure (Runtime checked position updated (Map.insert (boundAttempt bound) (Attempt bound Issued Nothing) attempts))

consume :: Consumption -> Runtime -> Either Error Runtime
consume reported runtime@(Runtime checked _ _ _) = do
    let bound = binding reported
    requirePhase Issued bound runtime
    unless (program reported == A.bytes checked) (Left ProgramMismatch)
    selected <- intent runtime (boundCall bound)
    unless (emission reported == selected) (Left (EmissionMismatch selected (emission reported)))
    pure (replaceAttempt (Attempt bound Consumed Nothing) runtime)

finish :: Binding -> ByteString -> Runtime -> Either Error Runtime
finish bound output runtime@(Runtime checked position _ _) = do
    requirePhase Consumed bound runtime
    Call assignments selected _ <- lookupCall runtime (boundCall bound)
    let completed =
            Completion
                { reportBinding = bound
                , reportProgram = A.bytes checked
                , reportCommand = position
                , reportInputs = assignments
                , reportEmission = selected
                , reportOutput = output
                }
    pure (replaceAttempt (Attempt bound Finished (Just completed)) runtime)

cancel :: Binding -> Runtime -> Either Error Runtime
cancel bound runtime@(Runtime checked position calls _) = do
    requirePhase Issued bound runtime
    Call assignments selected _ <- lookupCall runtime (boundCall bound)
    let Runtime _ _ _ updated = replaceAttempt (Attempt bound Cancelled Nothing) runtime
    pure (Runtime checked position (Map.insert (boundCall bound) (Call assignments selected Nothing) calls) updated)

intent :: Runtime -> CallId -> Either Error E.Emission
intent runtime name = intendedEmission <$> intention runtime name

intention :: Runtime -> CallId -> Either Error Intention
intention runtime@(Runtime checked position _ _) name = do
    Call assignments selected _ <- lookupCall runtime name
    pure (Intention (A.bytes checked) position assignments selected)

intendedProgram :: Intention -> ByteString
intendedProgram (Intention encoded _ _ _) = encoded

intendedCommand :: Intention -> Natural
intendedCommand (Intention _ position _ _) = position

intendedInputs :: Intention -> E.World
intendedInputs (Intention _ _ assignments _) = assignments

intendedEmission :: Intention -> E.Emission
intendedEmission (Intention _ _ _ selected) = selected

phase :: Runtime -> AttemptId -> Either Error Phase
phase runtime name = do
    Attempt _ current _ <- lookupAttempt runtime name
    pure current

completion :: Runtime -> AttemptId -> Either Error (Maybe Completion)
completion runtime name = do
    Attempt _ _ completed <- lookupAttempt runtime name
    pure completed

completedBinding :: Completion -> Binding
completedBinding = reportBinding

completedProgram :: Completion -> ByteString
completedProgram = reportProgram

completedCommand :: Completion -> Natural
completedCommand = reportCommand

completedInputs :: Completion -> E.World
completedInputs = reportInputs

completedEmission :: Completion -> E.Emission
completedEmission = reportEmission

completedOutput :: Completion -> ByteString
completedOutput = reportOutput

lookupCall :: Runtime -> CallId -> Either Error Call
lookupCall (Runtime _ _ calls _) name = maybe (Left (UnknownCall name)) Right (Map.lookup name calls)

lookupAttempt :: Runtime -> AttemptId -> Either Error Attempt
lookupAttempt (Runtime _ _ _ attempts) name = maybe (Left (UnknownAttempt name)) Right (Map.lookup name attempts)

requirePhase :: Phase -> Binding -> Runtime -> Either Error ()
requirePhase expected bound runtime = do
    Attempt actual current _ <- lookupAttempt runtime (boundAttempt bound)
    unless (actual == bound) (Left (BindingMismatch actual bound))
    unless (current == expected) (Left (PhaseMismatch expected current))

replaceAttempt :: Attempt -> Runtime -> Runtime
replaceAttempt attempt@(Attempt bound _ _) (Runtime checked position calls attempts) =
    Runtime checked position calls (Map.insert (boundAttempt bound) attempt attempts)
