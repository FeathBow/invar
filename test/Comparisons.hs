{-# LANGUAGE OverloadedStrings #-}

module Comparisons (comparisons) where

import Calls qualified
import Control.Monad (forM_)
import Data.Aeson (Value (..), object, toJSON, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.Text (Text)
import Data.Word (Word32)
import Hedgehog
import Invar.History.Cohort qualified as Cohort
import Invar.History.Generation qualified as Generation
import Invar.History.Runtime qualified as History
import Invar.History.Schedule qualified as Schedule
import Invar.History.Trace qualified as Trace
import Invar.Learn.Program qualified as Program
import Invar.Learn.Protocol qualified as Protocol
import Invar.Learn.Report qualified as Report
import Learning (taughtWorld)
import Recoveries qualified
import Traces qualified
import Updates qualified

comparisons :: Group
comparisons =
    Group
        "History schedules"
        [ ("a synchronous history and a runtime history of the same sessions, settings and learner records have one schedule", once normalized)
        , ("a retried runtime run keeps its schedule while its identities and recorded execution change", once retried)
        , ("histories run with one and with two sessions have one schedule", once permuted)
        , ("updates that differ in logical order, optimizer steps or staleness, and a missing generation, are named", once differing)
        , ("teacher call identities are checked at admission but do not change semantic input equality", once teacherCalls)
        , ("teacher declarations and scores remain part of semantic input equality", once teacherInputs)
        ]
  where
    once = withTests 1 . property

scheduleOf :: [Generation.Generation] -> Schedule.Schedule
scheduleOf generations = Schedule.project [Report.checkedRequest (Cohort.update (Generation.cohort generation)) | generation <- generations]

normalized :: PropertyT IO ()
normalized = do
    synchronous <- Traces.built 1 False >>= evalEither . Traces.admit
    runtime <- Recoveries.ran >>= Recoveries.finishedRun
    scheduleOf (Trace.generations synchronous) === scheduleOf (History.generations runtime)

retried :: PropertyT IO ()
retried = do
    fixture <- Recoveries.ran
    finished <- Recoveries.finishedRun fixture
    again <- Recoveries.retriedRun fixture
    scheduleOf (History.generations again) === scheduleOf (History.generations finished)
    History.identities again /== History.identities finished
    map (parseEither executed . History.recorded) [finished, again] === [Right (2, 1, 0), Right (4, 2, 1)]
  where
    executed :: Value -> Parser (Int, Int, Int)
    executed = withObject "recorded execution" $ \fields -> (,,) <$> fields .: "processes" <*> (length <$> (fields .: "attempts" :: Parser [Value])) <*> fields .: "restarts"

permuted :: PropertyT IO ()
permuted = do
    single <- Traces.built 1 False >>= evalEither . Traces.admit
    split <- Traces.built 2 False >>= evalEither . Traces.admit
    scheduleOf (Trace.generations single) === scheduleOf (Trace.generations split)

differing :: PropertyT IO ()
differing = do
    runtime <- Recoveries.ran >>= Recoveries.finishedRun
    report <- case History.generations runtime of
        [only] -> pure (Cohort.update (Generation.cohort only))
        _ -> failure
    original <- objectOf (Report.request report)
    returned <- objectOf (Report.result report)
    (bound, program) <- evalEither (parseEither (withObject "invocation" (\fields -> (,) <$> fields .: "binding" <*> fields .: "program")) (Report.invocation report))
    call <- evalEither (parseEither (withObject "binding" (.: "call")) bound)
    order <- evalEither (parseEither (.: "order") original)
    steps <- evalEither (parseEither (.: "steps") original)
    let admitted altered = Report.admit call (Calls.wire [object ["stage" .= String "consumed", "binding" .= bound, "program" .= (program :: Text), "request" .= altered], Object (Fields.insert "request" (Object altered) returned)])
        names = concat (steps :: [[Text]])
        changed = [("order", Fields.insert "order" (toJSON (reverse (order :: [Text]))) original), ("steps", Fields.insert "steps" (toJSON [take 2 names, drop 2 names]) original), ("staleness", Fields.insert "schedule" (object ["update" .= (0 :: Int), "staleness" .= (1 :: Int)]) original)]
    unchanged <- evalEither (admitted original)
    Schedule.differences (schedule [report]) (schedule [unchanged]) === []
    forM_ changed $ \(name, altered) -> do
        reparsed <- evalEither (admitted altered)
        Schedule.differences (schedule [report]) (schedule [reparsed]) === [object ["generation" .= (1 :: Int), "fields" .= [name :: Text]]]
    Schedule.differences (schedule []) (schedule [report]) === [object ["generation" .= (1 :: Int), "missing" .= ("left" :: Text)]]
  where
    schedule = Schedule.project . map Report.checkedRequest
    objectOf (Object fields) = pure fields
    objectOf _ = failure

teacherCalls :: PropertyT IO ()
teacherCalls = do
    (context, events) <- Updates.setupWith Program.distilled taughtWorld
    original <- evalEither (reported events)
    forM_ ["call", "attempt", "instance"] $ \axis -> do
        let renumber sample = Updates.change "teacher_call" (Updates.change axis (Number 99) (Updates.field "teacher_call" sample)) sample
        changed <- evalEither (teacherSample renumber (Report.request original))
        let renumbered = withRequest changed events
        compared <- evalEither (reported renumbered)
        Report.sameInput original compared === Right True
        Report.paired original compared === Right ()
        Report.request original /== Report.request compared
        case Updates.observe context (Updates.wire renumbered) of
            Left (Protocol.Mismatch problem) -> problem === "Actual update input differs from the checked lowering"
            unexpected -> annotateShow unexpected >> failure

teacherInputs :: PropertyT IO ()
teacherInputs = do
    (_, events) <- Updates.setupWith Program.distilled taughtWorld
    original <- evalEither (reported events)
    let input = Report.request original
        declaration = Updates.change "teacher" (Updates.change "revision" (String "another-revision") (Updates.field "teacher" input)) input
    scores <- evalEither (teacherSample (Updates.change "teacher_bits" (toJSON [0xbf800000 :: Word32])) input)
    forM_ [declaration, scores] $ \changed -> do
        compared <- evalEither (reported (withRequest changed events))
        Report.sameInput original compared === Right False

teacherSample :: (Value -> Value) -> Value -> Either String Value
teacherSample operation =
    parseEither
        ( withObject "request" $ \fields -> do
            samples <- fields .: "samples" :: Parser [Value]
            pure (Object (Fields.insert "samples" (toJSON (Updates.alter 0 operation samples)) fields))
        )

withRequest :: Value -> [Value] -> [Value]
withRequest input = Updates.alter 2 (Updates.change "request" input) . Updates.alter 1 (Updates.change "request" input)

reported :: [Value] -> Either String Report.Report
reported events = do
    call <- parseEither (withObject "update binding" (.: "call")) Updates.bound
    Report.admit call (Updates.wire events)
