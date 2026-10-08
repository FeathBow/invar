{-# LANGUAGE OverloadedStrings #-}

module Comparisons (comparisons) where

import Calls qualified
import Control.Monad (forM_)
import Data.Aeson (Value (..), object, toJSON, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.Text (Text)
import Hedgehog
import Invar.History.Cohort qualified as Cohort
import Invar.History.Generation qualified as Generation
import Invar.History.Runtime qualified as History
import Invar.History.Schedule qualified as Schedule
import Invar.History.Trace qualified as Trace
import Invar.Learn.Report qualified as Report
import Recoveries qualified
import Traces qualified

comparisons :: Group
comparisons =
    Group
        "History schedules"
        [ ("a synchronous history and a runtime history of the same sessions, settings and learner records have one schedule", once normalized)
        , ("a retried runtime run keeps its schedule while its identities and recorded execution change", once retried)
        , ("histories run with one and with two sessions have one schedule", once permuted)
        , ("updates that differ in logical order, optimizer steps or staleness, and a missing generation, are named", once differing)
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
