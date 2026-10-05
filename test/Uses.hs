{-# LANGUAGE OverloadedStrings #-}

module Uses (uses) where

import Calls (change, field, wire)
import Control.Monad (forM_)
import Data.Aeson (Value (..), eitherDecodeStrict, encode)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.List.NonEmpty qualified as NonEmpty
import Data.Ratio ((%))
import Hedgehog
import Invar.Numerical qualified as N
import Invar.Use qualified as U
import Invar.Use.Decimal qualified as Decimal
import Invar.Workload qualified as Workload
import UseFixture qualified as F

uses :: Group
uses =
    Group
        "Checked paired task-loss observations"
        [ ("losses rescore complete responses and weight prompts equally", once weighting)
        , ("case delivery order cannot change scope or weighting", once ordering)
        , ("truncated correct-looking answers retain loss one", once truncation)
        , ("complete inventory rejects missing duplicate and unexpected cases", once inventory)
        , ("the same prompt cannot inflate independent units or repeat seeds", once questions)
        , ("workload answers and source records are part of the scope", once scopes)
        , ("declared task inputs must match admitted generation", once correspondence)
        , ("unsuccessful malformed and reused executions fail observation", once executions)
        , ("a batch cannot silently change one implementation", once implementation)
        , ("Unicode prompts remain distinct and retain exact loss scopes", once unicode)
        ]
  where
    once = withTests 1 . property

weighting :: PropertyT IO ()
weighting = do
    supplied <- F.fixture F.trials
    observed <- evalEither (U.observe supplied)
    length (U.units observed) === 2
    length (U.numerical observed) === 4
    U.mean U.ReferenceLoss observed === Just (1 % 2)
    U.mean U.CandidateLoss observed === Just (2 % 3)
    U.mean U.LossIncrease observed === Just (1 % 6)
    map (length . U.members) (NonEmpty.toList (U.units observed)) === [3, 1]
    field "unit_count" (U.describe observed) === Number 2
    field "sample_count" (U.describe observed) === Number 4

ordering :: PropertyT IO ()
ordering = do
    supplied <- F.fixture F.trials
    U.observe supplied === U.observe supplied {U.cases = reverse (U.cases supplied)}

truncation :: PropertyT IO ()
truncation = do
    supplied <- F.fixture [trial {F.truncated = True, F.after = "#### 12"} | trial <- F.trials]
    observed <- evalEither (U.observe supplied)
    U.mean U.ReferenceLoss observed === Just 1
    U.mean U.CandidateLoss observed === Just 1
    U.mean U.LossIncrease observed === Just 0

inventory :: PropertyT IO ()
inventory = do
    supplied <- F.fixture F.trials
    case U.cases supplied of
        first : remaining -> do
            U.observe supplied {U.cases = first : first : remaining} === Left (U.DuplicateCase (U.caseKey first))
            rejected (U.observe supplied {U.cases = remaining})
            rejected (U.observe supplied {U.cases = first {U.caseKey = U.Key 9 "missing"} : remaining})
        [] -> failure

questions :: PropertyT IO ()
questions = do
    forM_ [\trial -> trial {F.answer = "#### 9"}, \trial -> trial {F.seed = 1}] $ \alter -> do
        let values = case F.trials of a : b : rest -> a : alter b : rest; rest -> rest
        document <- evalEither (Workload.decode (Lazy.toStrict (encode (F.workloadValue values))))
        case Decimal.domain document of
            Left _ -> success
            Right _ -> failure

scopes :: PropertyT IO ()
scopes = do
    supplied <- F.fixture F.trials
    before <- evalEither (U.observe supplied)
    document <- evalEither (Workload.decode (Lazy.toStrict (encode (F.workloadValue F.trials)) <> "\n"))
    declared <- evalEither (Decimal.domain document)
    after <- evalEither (U.observe supplied {U.domain = declared})
    assert (U.scope before /= U.scope after)
    assert (U.scopeId (U.scope before) /= U.scopeId (U.scope after))
    U.mean U.LossIncrease before === U.mean U.LossIncrease after
    changed <- F.fixture [trial {F.answer = "#### 0"} | trial <- F.trials] >>= evalEither . U.observe
    assert (U.scope before /= U.scope changed)
    U.mean U.LossIncrease changed === Just ((-1) % 6)

correspondence :: PropertyT IO ()
correspondence = do
    supplied <- F.fixture F.trials
    let variants = [\trial -> trial {F.prompt = "other " ++ F.prompt trial}, \trial -> trial {F.seed = F.seed trial + 99}]
    forM_ variants $ \alter -> do
        changed <- F.fixture (map alter F.trials)
        rejected (U.observe supplied {U.domain = U.domain changed})

executions :: PropertyT IO ()
executions = do
    supplied <- F.fixture F.trials
    forM_ [\run -> run {N.exitCode = 7}, \run -> run {N.logBytes = Bytes.init (N.logBytes run)}] $ \alter ->
        rejected (U.observe supplied {U.cases = map (changeCandidate alter) (U.cases supplied)})
    repeated <- traverse repeatBinding (zip (U.cases supplied) F.trials)
    case U.observe supplied {U.cases = repeated} of
        Left (U.ReusedExecution N.Candidate _) -> success
        other -> annotateShow other >> failure
  where
    repeatBinding (supplied, trial) = do
        candidate <- F.run 1 N.Candidate trial
        pure supplied {U.paired = N.BoundRun (N.reference (U.paired supplied)) candidate}

implementation :: PropertyT IO ()
implementation = do
    supplied <- F.fixture F.trials
    case U.cases supplied of
        first : remaining -> do
            let run = N.candidate (U.paired first)
            events <- evalEither (traverse eitherDecodeStrict (Bytes.lines (N.logBytes run)))
            let changed = [if field "stage" event == String "loaded_adapter" then change "revision" (String "other-model-revision") event else event | event <- events]
                candidate = run {N.logBytes = wire changed}
                pair = N.BoundRun (N.reference (U.paired first)) candidate
            case U.observe supplied {U.cases = first {U.paired = pair} : remaining} of
                Left (U.MixedImplementation N.Candidate _) -> success
                other -> annotateShow other >> failure
        [] -> failure

changeCandidate :: (N.Run -> N.Run) -> U.Case -> U.Case
changeCandidate alter value = value {U.paired = N.BoundRun (N.reference pair) (alter (N.candidate pair))}
  where
    pair = U.paired value

unicode :: PropertyT IO ()
unicode = do
    let rename trial = trial {F.prompt = if F.prompt trial == "question a" then "题目甲" else "题目乙"}
    observed <- F.fixture (map rename F.trials) >>= evalEither . U.observe
    length (U.units observed) === 2
    U.mean U.LossIncrease observed === Just (1 % 6)
    case field "judgement" (U.describeFinding (U.establish (U.Claim (U.scope observed) U.LossIncrease 1) observed)) of
        Object _ -> success
        other -> annotateShow other >> failure

rejected :: (Show value) => Either U.ObservationError value -> PropertyT IO ()
rejected (Left _) = success
rejected (Right value) = annotateShow value >> failure
