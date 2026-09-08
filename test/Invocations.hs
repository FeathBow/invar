{-# LANGUAGE OverloadedStrings #-}

module Invocations (invocations) where

import Control.Monad (forM_)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Invocation qualified as I
import Invar.Spec.Program
import Invar.Spec.Value (Scalar (..), Value (..))
import Properties (campaign)

invocations :: Group
invocations =
    Group
        "Invocation reports"
        [ ("intent consumption and completion remain distinct", once lifecycle)
        , ("consumption binds program sink specification and payload", once payloadBinding)
        , ("reports bind the call attempt and instance", once reportBinding)
        , ("identifiers and terminal reports cannot be reused", once duplicates)
        , ("acknowledged non-consumption permits a fresh attempt", once retry)
        , ("equal payloads remain distinct logical occurrences", once occurrences)
        , ("preparation validates inputs and command positions", once preparation)
        , ("operational inputs do not choose the bound emission", campaign histories)
        ]
  where
    once = withTests 1 . property

program :: Rational -> Either A.LoadError A.Checked
program constant = A.load (A.encode meaning commands)
  where
    allowed = Set.singleton (Semantic "value")
    schema = Schema (Map.fromList [(Semantic "value", NumberType), (Operational "history", BooleanType)]) Map.empty (Map.singleton "out" (Sink "reference" NumberType allowed Set.empty))
    meaning = E.Semantics schema Map.empty
    commands = [Emit "out" "reference" (Constant NumberType (Atom (Number constant))), Emit "out" "reference" (Read (Input (Semantic "value")))]

world :: Rational -> Bool -> E.World
world value history = Map.fromList [(Semantic "value", Atom (Number value)), (Operational "history", Atom (Boolean history))]

bound :: I.Binding
bound = I.Binding (I.CallId 0) (I.AttemptId 0) (I.Instance 0)

expected :: Rational -> E.Emission
expected = E.Emission "out" "reference" . Atom . Number

prepared :: A.Checked -> Either I.Error I.Runtime
prepared checked = I.prepare (I.Selection (I.boundCall bound) (world 3 False)) (I.start checked 1)

issued :: A.Checked -> Either I.Error I.Runtime
issued checked = prepared checked >>= I.issue bound

consumption :: A.Checked -> I.Consumption
consumption checked = I.Consumption bound (A.bytes checked) (expected 3)

reject :: I.Error -> Either I.Error I.Runtime -> PropertyT IO ()
reject expectedError result = case result of
    Left actual -> actual === expectedError
    Right _ -> failure

lifecycle :: PropertyT IO ()
lifecycle = do
    checked <- evalEither (program 0)
    ready <- evalEither (prepared checked)
    I.intent ready (I.boundCall bound) === Right (expected 3)
    sent <- evalEither (I.issue bound ready)
    I.phase sent (I.boundAttempt bound) === Right I.Issued
    I.completion sent (I.boundAttempt bound) === Right Nothing
    reject (I.PhaseMismatch I.Consumed I.Issued) (I.finish bound "output" sent)
    used <- evalEither (I.consume (consumption checked) sent)
    I.phase used (I.boundAttempt bound) === Right I.Consumed
    I.completion used (I.boundAttempt bound) === Right Nothing
    reject (I.PhaseMismatch I.Issued I.Consumed) (I.cancel bound used)
    done <- evalEither (I.finish bound "output" used)
    I.phase done (I.boundAttempt bound) === Right I.Finished
    result <- evalEither (I.completion done (I.boundAttempt bound)) >>= evalMaybe
    I.completedBinding result === bound
    I.completedProgram result === A.bytes checked
    I.completedCommand result === 1
    I.completedInputs result === world 3 False
    I.completedEmission result === expected 3
    I.completedOutput result === "output"

payloadBinding :: PropertyT IO ()
payloadBinding = do
    checked <- evalEither (program 0)
    changed <- evalEither (program 1)
    sent <- evalEither (issued checked)
    let report = consumption checked
    reject I.ProgramMismatch (I.consume report {I.program = A.bytes changed} sent)
    forM_ [expected 4, E.Emission "other" "reference" (Atom (Number 3)), E.Emission "out" "other" (Atom (Number 3))] $ \wrong ->
        reject (I.EmissionMismatch (expected 3) wrong) (I.consume report {I.emission = wrong} sent)
    _ <- evalEither (I.consume report sent)
    pure ()

reportBinding :: PropertyT IO ()
reportBinding = do
    checked <- evalEither (program 0)
    sent <- evalEither (issued checked)
    used <- evalEither (I.consume (consumption checked) sent)
    forM_ [bound {I.boundCall = I.CallId 1}, bound {I.boundInstance = I.Instance 1}] $ \wrong -> do
        reject (I.BindingMismatch bound wrong) (I.consume (consumption checked) {I.binding = wrong} sent)
        reject (I.BindingMismatch bound wrong) (I.finish wrong "output" used)
    let missing = bound {I.boundAttempt = I.AttemptId 1}
    reject (I.UnknownAttempt (I.AttemptId 1)) (I.consume (consumption checked) {I.binding = missing} sent)
    reject (I.UnknownAttempt (I.AttemptId 1)) (I.finish missing "output" used)

duplicates :: PropertyT IO ()
duplicates = do
    checked <- evalEither (program 0)
    sent <- evalEither (issued checked)
    reject (I.DuplicateCall (I.CallId 0)) (I.prepare (I.Selection (I.CallId 0) (world 9 True)) sent)
    reject (I.CallInUse (I.CallId 0) (I.AttemptId 0)) (I.issue bound {I.boundAttempt = I.AttemptId 1} sent)
    used <- evalEither (I.consume (consumption checked) sent)
    reject (I.PhaseMismatch I.Issued I.Consumed) (I.consume (consumption checked) used)
    done <- evalEither (I.finish bound "first" used)
    reject (I.PhaseMismatch I.Consumed I.Finished) (I.finish bound "replacement" done)
    reject (I.CallInUse (I.CallId 0) (I.AttemptId 0)) (I.issue bound {I.boundAttempt = I.AttemptId 1} done)
    result <- evalEither (I.completion done (I.AttemptId 0)) >>= evalMaybe
    I.completedOutput result === "first"

retry :: PropertyT IO ()
retry = do
    checked <- evalEither (program 0)
    sent <- evalEither (issued checked)
    cancelled <- evalEither (I.cancel bound sent)
    I.phase cancelled (I.AttemptId 0) === Right I.Cancelled
    reject (I.DuplicateAttempt (I.AttemptId 0)) (I.issue bound cancelled)
    reject (I.PhaseMismatch I.Issued I.Cancelled) (I.cancel bound cancelled)
    let next = bound {I.boundAttempt = I.AttemptId 1, I.boundInstance = I.Instance 1}
    resent <- evalEither (I.issue next cancelled)
    reject (I.PhaseMismatch I.Issued I.Cancelled) (I.consume (consumption checked) resent)
    reject (I.PhaseMismatch I.Consumed I.Cancelled) (I.finish bound "late" resent)
    used <- evalEither (I.consume (consumption checked) {I.binding = next} resent)
    done <- evalEither (I.finish next "fresh" used)
    I.completion done (I.AttemptId 0) === Right Nothing
    result <- evalEither (I.completion done (I.AttemptId 1)) >>= evalMaybe
    I.completedBinding result === next
    I.completedOutput result === "fresh"

occurrences :: PropertyT IO ()
occurrences = do
    checked <- evalEither (program 0)
    sent <- evalEither (issued checked)
    ready <- evalEither (I.prepare (I.Selection (I.CallId 1) (world 3 True)) sent)
    let reused = bound {I.boundCall = I.CallId 1}
        second = reused {I.boundAttempt = I.AttemptId 1}
    reject (I.DuplicateAttempt (I.AttemptId 0)) (I.issue reused ready)
    both <- evalEither (I.issue second ready)
    used <- evalEither (I.consume (consumption checked) {I.binding = second} both)
    done <- evalEither (I.finish second "second" used)
    I.phase done (I.AttemptId 0) === Right I.Issued
    I.completion done (I.AttemptId 0) === Right Nothing
    I.phase done (I.AttemptId 1) === Right I.Finished

preparation :: PropertyT IO ()
preparation = do
    checked <- evalEither (program 0)
    let runtime = I.start checked 1
    reject (I.UnknownCall (I.CallId 0)) (I.issue bound runtime)
    reject (I.MissingCommand 2) (I.prepare (I.Selection (I.CallId 0) (world 3 False)) (I.start checked 2))
    reject (I.InvalidInputs (E.MissingInput (Semantic "value"))) (I.prepare (I.Selection (I.CallId 0) Map.empty) runtime)
    first <- evalEither (I.prepare (I.Selection (I.CallId 0) (world 3 False)) (I.start checked 0))
    I.intent first (I.CallId 0) === Right (expected 0)

histories :: PropertyT IO ()
histories = do
    value <- fromInteger <$> forAll (Gen.integral (Range.linear (-magnitude) magnitude))
    history <- forAll Gen.bool
    checked <- evalEither (program 0)
    forM_ [history, not history] $ \operational -> do
        ready <- evalEither (I.prepare (I.Selection (I.CallId 0) (world value operational)) (I.start checked 1))
        I.intent ready (I.CallId 0) === Right (expected value)
  where
    magnitude = 10
