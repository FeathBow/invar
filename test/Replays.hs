{-# LANGUAGE OverloadedStrings #-}

module Replays (replays) where

import Calls qualified
import Control.Monad (forM_)
import Data.Aeson (Object, Value (..), eitherDecodeStrict, encode, object, toJSON, (.:), (.=))
import Data.Aeson.Key qualified as Key
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Either (isLeft)
import Data.Foldable (toList)
import Data.IORef (atomicModifyIORef', newIORef, readIORef)
import Data.List (isInfixOf, partition)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8)
import Hedgehog hiding (Update)
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Async.Completion qualified as Completion
import Invar.Async.Core (Attempt (..), Phase (..))
import Invar.Async.Core qualified as Core
import Invar.Async.Entry qualified as Entry
import Invar.Async.Plan (Declared (Declared), Request (..), Update (..))
import Invar.Async.Plan qualified as Plan
import Invar.Async.Replay qualified as Replay
import Invar.Cohort qualified as C
import Invar.History.Generation qualified as Generation
import Invar.Infer.Replay qualified as Logged
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Journal qualified as Journal
import Invar.Learn qualified as L
import Invar.Learn.Protocol qualified as Protocol
import Invar.Learn.Report qualified as Report
import Invar.Learn.Stream qualified as S
import Invar.Learn.Worker qualified as Learner
import Invar.Loop qualified as Loop
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as R
import Invar.Spec.Invocation qualified as V
import Invar.Store qualified as Store
import Invar.Transcript qualified as Transcript
import Invar.Worker qualified as Worker
import LearnerFixture qualified as F
import Numeric.Natural (Natural)
import ResidentFixture qualified as Resident
import ResidentWorkloads qualified as Workloads
import Sessions qualified
import Store (workspace)
import System.Exit (ExitCode (..))
import System.FilePath ((</>))

replays :: Group
replays =
    Group
        "Runtime journal replay"
        [ ("every runtime entry reads back as the entry that was written", withTests 300 (property typed))
        , ("an entry of another format, with a field it does not declare or without one it needs is refused", once strict)
        , ("a journal of finite or resident inference and a process or resident learner replays its transcripts to the published generation, the result of each call and the identity floors", once replayed)
        , ("a resident owner's releases and close are admitted only by their acknowledgements", once owners)
        , ("a reserved process without a transcript is recovered unless a later entry names it, and a resident owner without calls still closes", once reserved)
        , ("an attempt under a number the core did not assign is refused, also at the end of a journal", once assigned)
        , ("a completion or learner event repeated where it was journaled changes nothing, as in the core", once repeated)
        , ("a call on a process of another role, slot or epoch, on an ended process or under a used identity is refused", once misplaced)
        , ("a changed transcript line, a different stored digest, a different applied record and a result never completed are told apart", once altered)
        , ("a readiness, step, staged adapter or verified learner that differs from the learner's transcript is refused", once claims)
        , ("a finite session yields its results only after a journaled clean exit", once exits)
        , ("a learner result followed by an unreadable line or a fragment, naming another call or request, or after an invalid measurement supports no publication while its steps stay admitted", once results)
        , ("a learner result supports a publication only after its process exited cleanly or its resident release was acknowledged", once boundaries)
        , ("after a restart a request is started or completed only through a call dispatched since, on the worker and epoch of that call's process", once redispatched)
        , ("a resident owner that exits cleanly before its results are stored leaves them admitted", once drained)
        , ("a learner transcript cut inside an attempt keeps its unconfirmed tail out of the core", once cut)
        , ("an attempt is committed when journaled or confirmed by a restart, concluded when published nowhere, and incomplete before its result; committed evidence holds what the attempt consumed", once outcomes)
        , ("a generation is built only from one learner admission, its own cohort and the version its schedule names", once generations)
        , ("a restart publishes the recording attempt a matching generation confirms, redoes it when none is observed and refuses any other generation", once restarts)
        ]
  where
    once = withTests 1 . property

data Run = Run {declaration :: Replay.Declaration, entries :: [Entry.Entry], transcripts :: Map Natural ByteString, records :: [Value], published :: Entry.Generation, members :: Natural, learnerProcess :: Natural, tasks :: [C.Task]}

described :: PropertyT IO Policy.Description
described = evalEither (Policy.describe ("test-model", "test-revision") (L.policy settings, L.tokenizer settings, L.behaviorBase settings, L.behaviorAssembly settings))
  where
    settings = F.configured

built :: R.Mode -> Learner.Mode -> PropertyT IO Run
built inference learning = do
    root <- workspace
    chosen <- if inference == R.Resident then Workloads.initial <$> Workloads.setup root 1 [L.policy F.configured] else Sessions.options root 1
    initial <- described
    let settings = F.configured
        cycle' = Loop.Cycle (C.tasks (R.definition chosen)) (R.order chosen) (R.delivery chosen)
        count = fromIntegral (length (Loop.tasks cycle'))
        bound = V.ordinal count
        resident = learning == Learner.Resident
    bound' <- evalEither (first show (Loop.bindTasks initial (Loop.tasks cycle')))
    journaled <- evalIO (newIORef [])
    captured <- evalIO (newIORef Map.empty)
    numbers <- evalIO (newIORef 0)
    slots <- evalIO (newIORef Map.empty)
    let append recorded = atomicModifyIORef' journaled (\held -> (held ++ recorded, ()))
        written number bytes = atomicModifyIORef' captured (\held -> (Map.insertWith (flip (<>)) number bytes held, ()))
        reserve role slot = do
            number <- atomicModifyIORef' numbers (\next -> (next + 1, next))
            append [Entry.Reserved number role slot 0]
            pure number
        opened slot = do
            number <- reserve Entry.Inference slot
            atomicModifyIORef' slots (\held -> (Map.insert slot number held, ()))
            pure (Transcript.Transcript (written number . (<> "\n")) (written number) (append . pure . Entry.Finished number))
        dispatched slot requests = do
            number <- (Map.! slot) <$> readIORef slots
            append ([Entry.Dispatched index slot 0 call number | (index, call) <- requests] ++ [Entry.Happened (Entry.Started slot 0 index) | (index, _) <- requests])
        checked index admitted = append [Entry.Stored index (Trajectory.binding admitted) (Trajectory.digest admitted), Entry.Happened (Entry.Completed 0 0 index (Trajectory.digest admitted))]
    produced <- evalIO $ R.withRecordedDriver inference (R.worker chosen, R.sessions chosen) opened $ \driver -> do
        owned <- if resident then Just <$> reserve Entry.Learner 0 else pure Nothing
        append [Entry.Opened 0 0, Entry.Happened (Entry.Connected 0 0)]
        batch <- R.runObserved driver (R.Observer dispatched checked) chosen {R.definition = C.Definition (L.policy settings) bound'} >>= F.require
        planned <- F.require (L.prepare settings batch)
        number <- maybe (reserve Entry.Learner 0) pure owned
        exchange <- F.prepare root (0, bound) planned
        pure (number, F.before exchange, F.steps exchange, F.after exchange, F.released exchange)
    (number, before, steps, after, released) <- evalEither produced
    rolled <- evalIO (readIORef journaled)
    sessions <- evalIO (readIORef captured)
    let prefix = if resident then before else filter ((/= Just "activation") . F.stage) before
        closed = object ["stage" .= String "closed", "format" .= String "invar-resident-v1", "owner" .= F.owner, "groups" .= (1 :: Int), "measurement" .= decodeUtf8 (F.wire [F.timer "closed"])]
        learner' = prefix ++ steps ++ after ++ (if resident then [released, closed] else [])
        loading = map Just ["loading", "profile", "load", "activation"]
    report <- evalEither (Report.admit count (F.wire (dropWhile ((`elem` loading) . F.stage) (prefix ++ steps ++ after))))
    reported <- traverse fieldsOf steps
    (begun, _) <- evalEither (first show (Protocol.partial bound (Report.checkedRequest report) []))
    (stream, stopped) <- evalEither (first show (Protocol.partial bound (Report.checkedRequest report) reported))
    assert (null stopped)
    finished <- fieldsOf (Report.result report)
    adapter <- evalEither (parseEither (.: "adapter") finished)
    learner <- evalEither (parseEither (.: "learner") finished)
    successor <- evalEither (Policy.successor adapter initial)
    let worker = R.worker chosen
        backend = Loop.Backend "unused learning executable" (Worker.executable worker) (Worker.script worker) (Worker.configuration worker) inference "unused update script" learning (Worker.cache worker) (R.sessions chosen)
        config = Loop.Config backend (root </> "run") (root </> "input") (root </> "reference") settings Store.RenameExclusive
        requests = map Request [0 .. count - 1]
        exited = Entry.Finished number (Transcript.Exited ExitSuccess Transcript.Complete)
        attempt =
            [Entry.Attempted 0 0 bound number, Entry.Happened (Entry.Ready 0 0 bound (S.identity stream) (S.opening begun))]
                ++ concat [[Entry.Happened (Entry.Current 0 0 (Completion.step done) (Completion.before done)), Entry.Happened (Entry.claim (Core.Applied (Update 0) (Attempt 0) done))] | done <- S.completions stream]
                ++ [exited | not resident]
                ++ [Entry.Verified 0 0 learner, Entry.Happened (Entry.Staged 0 0 adapter), Entry.Happened (Entry.Recorded 0 0), Entry.Happened (Entry.Committed 0 0)]
                ++ [exited | resident]
        (closes, kept) = partition (\recorded -> inference == R.Resident && isFinished recorded) rolled
    planned <- evalEither (first show (Plan.prepare 0 [Declared requests [requests]]))
    pure
        Run
            { declaration = Replay.Declaration config 0 planned [const (Right cycle')] [0] initial
            , entries = kept ++ attempt ++ closes
            , transcripts = Map.insert number (F.wire learner') sessions
            , records = learner'
            , published = Entry.Generation 1 adapter learner successor
            , members = count
            , learnerProcess = number
            , tasks = bound'
            }
  where
    isFinished Entry.Finished {} = True
    isFinished _ = False

serial :: PropertyT IO Run
serial = built R.Serial Learner.Process

fieldsOf :: Value -> PropertyT IO Object
fieldsOf (Object fields) = pure fields
fieldsOf _ = failure

replayOf :: Run -> Either String Replay.Replayed
replayOf run = Replay.replay (declaration run) (entries run) (transcripts run)

through :: Entry.Entry -> [Entry.Entry] -> [Entry.Entry]
through last' journal = let (leading, rest) = break (== last') journal in leading ++ take 1 rest

firstOf :: [value] -> PropertyT IO value
firstOf (chosen : _) = pure chosen
firstOf [] = failure

refused :: String -> Either String value -> PropertyT IO ()
refused expected outcome = case outcome of
    Left problem -> do
        annotate problem
        assert (expected `isInfixOf` problem)
    Right _ -> failure

typed :: PropertyT IO ()
typed = do
    initial <- described
    written <- forAll (generated initial)
    decoded <- evalEither (Journal.entries (Lazy.toStrict (encode (Entry.encode written)) <> "\n"))
    traverse (parseEither Entry.decode) decoded === Right [written]
  where
    number = Gen.integral (Range.linear 0 1000) :: Gen Natural
    text = Gen.string (Range.linear 0 12) Gen.alphaNum
    bound = V.Binding . V.CallId <$> number <*> (V.AttemptId <$> number) <*> (V.Instance <$> number)
    claim =
        Gen.choice
            [ Entry.Connected <$> number <*> number
            , Entry.Lost <$> number <*> number
            , Entry.Started <$> number <*> number <*> number
            , Entry.Completed <$> number <*> number <*> number <*> text
            , Entry.Ready <$> number <*> number <*> bound <*> text <*> text
            , Entry.Current <$> number <*> number <*> number <*> text
            , Entry.Applied <$> number <*> number <*> bound <*> text <*> number <*> text <*> text <*> text
            , Entry.Staged <$> number <*> number <*> text
            , Entry.Recorded <$> number <*> number
            , Entry.Committed <$> number <*> number
            , Entry.Abandoned <$> number <*> number
            ]
    read' = Gen.element [Transcript.Complete, Transcript.Cut]
    ended = Gen.choice [Transcript.Unlaunched <$> text, Transcript.Exited ExitSuccess <$> read', Transcript.Exited . ExitFailure <$> Gen.filter (/= 0) (Gen.int (Range.linear (-64) 255)) <*> read', Transcript.Stopped <$> read']
    declared = Fields.fromList <$> Gen.list (Range.linear 0 4) ((,) . Key.fromString <$> Gen.filter (`notElem` ["entry", "format", "directory"]) text <*> (toJSON <$> text))
    generated initial =
        Gen.choice
            [ Entry.Declared <$> text <*> declared
            , Entry.Opened <$> number <*> number
            , Entry.Dispatched <$> number <*> number <*> number <*> bound <*> number
            , Entry.Attempted <$> number <*> number <*> bound <*> number
            , Entry.Reserved <$> number <*> Gen.element [Entry.Inference, Entry.Learner] <*> number <*> number
            , Entry.Finished <$> number <*> ended
            , Entry.Happened <$> claim
            , Entry.Stored <$> number <*> bound <*> text
            , Entry.Verified <$> number <*> number <*> text
            , Entry.Restarted <$> Gen.list (Range.linear 0 3) (Entry.Generation <$> number <*> text <*> text <*> pure initial)
            , Entry.Elapsed <$> Gen.element ["rollout", "learner"] <*> number <*> Gen.double (Range.linearFrac 0 1.0e6) <*> Gen.double (Range.linearFrac 0 1.0e6)
            ]

strict :: PropertyT IO ()
strict = do
    initial <- described
    let decoded value = Journal.entries (Lazy.toStrict (encode value) <> "\n") >>= traverse (parseEither Entry.decode)
        dispatched = Entry.encode (Entry.Dispatched 0 0 0 (V.ordinal 0) 0)
        restart = Entry.encode (Entry.Restarted [Entry.Generation 1 "adapter" "learner" initial])
        started = Entry.encode (Entry.Happened (Entry.Started 0 0 0))
        nested name change value = Calls.change name (change (Calls.field name value)) value
        without name (Object fields) = Object (Fields.delete name fields)
        without _ value = value
    _ <- evalEither (decoded dispatched)
    _ <- evalEither (decoded restart)
    forM_
        [ Calls.change "format" (String "invar-runtime-journal-v1") (Entry.encode (Entry.Declared "." Fields.empty))
        , without "format" (Entry.encode (Entry.Declared "." Fields.empty))
        , without "process" dispatched
        , nested "binding" (Calls.change "extra" (toJSON (0 :: Int))) dispatched
        , Calls.change "commands" (toJSON ([] :: [Value])) started
        , nested "event" (Calls.change "extra" (toJSON (0 :: Int))) started
        , Calls.change "generations" (toJSON [Calls.change "extra" (toJSON (0 :: Int)) generation | Array listed <- [Calls.field "generations" restart], generation <- toList listed]) restart
        , Calls.change "status" (toJSON (0 :: Int)) (Entry.encode (Entry.Finished 0 (Transcript.Stopped Transcript.Cut)))
        ]
        $ \value -> do
            annotateShow value
            assert (isLeft (decoded value))

replayed :: PropertyT IO ()
replayed = forM_ [(R.Serial, Learner.Process), (R.Serial, Learner.Resident), (R.Resident, Learner.Process), (R.Resident, Learner.Resident)] $ \(inference, learning) -> do
    run <- built inference learning
    result <- evalEither (replayOf run)
    let generation = published run
    Core.committed (Replay.state result) === [Update 0]
    Core.learning (Replay.state result) === Nothing
    Map.lookup 1 (Replay.versions result) === Just (Entry.adapter generation, Entry.learner generation, Entry.description generation)
    [request | (request, Replay.Completed _) <- Map.elems (Replay.statuses result)] === [0 .. members run - 1]
    case Map.toList (Replay.learned result) of
        [((0, 0), learned)] -> do
            Replay.binding learned === V.ordinal (members run)
            Replay.process learned === learnerProcess run
            Replay.confirmed learned === fromIntegral (length (Replay.completions learned))
            Replay.outcome learned === Replay.Committed 1
            Replay.consumed learned === map V.ordinal [0 .. members run - 1]
            evidenced <- evalMaybe (Map.lookup 1 (Replay.evidence result))
            Replay.committing evidenced === (0, 0)
            map (Trajectory.binding . Logged.trajectory) (Replay.trajectories evidenced) === Replay.consumed learned
            (Report.artifact "adapter" (Replay.report evidenced), Report.artifact "learner" (Replay.report evidenced)) === (Right (Entry.adapter generation), Right (Entry.learner generation))
            direct <- evalEither (Report.admitFrames (members run) (Replay.frames evidenced))
            let fields reported = (Report.invocation reported, Report.request reported, Report.result reported, Report.gradient reported)
            fields (Replay.report evidenced) === fields direct
        _ -> failure
    Replay.floors result === Replay.Floors (members run + 1) 2 1 1

owners :: PropertyT IO ()
owners = do
    run <- built R.Resident Learner.Resident
    _ <- evalEither (replayOf run)
    let learner' = learnerProcess run
        rewritten number change = run {transcripts = Map.adjust change number (transcripts run)}
        unclosed = Bytes.unlines . reverse . drop 1 . reverse . Bytes.lines
        stopped number = map (\recorded -> if recorded == Entry.Finished number (Transcript.Exited ExitSuccess Transcript.Complete) then Entry.Finished number (Transcript.Stopped Transcript.Cut) else recorded)
        acknowledged = F.wire [if F.stage record == Just "released" then Calls.change "result_sha256" (String (Text.replicate 64 "0")) record else record | record <- records run]
    refused "acknowledgement differs" (replayOf run {transcripts = Map.insert learner' acknowledged (transcripts run)})
    forM_ [0, learner'] $ \number -> forM_ [unclosed, (<> "not a record\n"), (<> "{\"stage\"")] $ \change -> do
        refused "close acknowledgement" (replayOf (rewritten number change))
        _ <- evalEither (replayOf (rewritten number change) {entries = stopped number (entries run)})
        pure ()

reserved :: PropertyT IO ()
reserved = do
    run <- serial
    let pending = through (Entry.Reserved 1 Entry.Learner 0 0) (entries run)
        missing = Map.delete 1 (transcripts run)
    result <- evalEither (Replay.replay (declaration run) pending missing)
    Replay.floors result === Replay.Floors (members run) 2 1 1
    Map.size (Replay.learned result) === 0
    refused "transcript is missing" (Replay.replay (declaration run) (through (Entry.Attempted 0 0 (V.ordinal (members run)) 1) (entries run)) missing)
    forM_ [Transcript.Unlaunched "never started", Transcript.Stopped Transcript.Cut] $ \outcome ->
        refused "transcript is missing" (Replay.replay (declaration run) (pending ++ [Entry.Finished 1 outcome]) missing)
    pool <- built R.Resident Learner.Process
    let idle = entries pool ++ [Entry.Reserved 9 Entry.Inference 1 0, Entry.Finished 9 (Transcript.Exited ExitSuccess Transcript.Complete)]
    _ <- evalEither (Replay.replay (declaration pool) idle (Map.insert 9 (F.wire [Resident.closed (Resident.scenarioWith 1 [])]) (transcripts pool)))
    refused "close acknowledgement" (Replay.replay (declaration pool) idle (Map.insert 9 "" (transcripts pool)))

assigned :: PropertyT IO ()
assigned = do
    run <- serial
    let attempted = Entry.Attempted 0 0 (V.ordinal (members run)) (learnerProcess run)
        prefix = through attempted (entries run)
    result <- evalEither (Replay.replay (declaration run) prefix (transcripts run))
    Replay.attempt (Replay.floors result) === 1
    forM_ [Entry.Attempted 0 999 (V.ordinal (members run)) (learnerProcess run), Entry.Attempted 1 1 (V.ordinal (members run)) (learnerProcess run)] $ \other ->
        refused "did not assign" (Replay.replay (declaration run) (takeWhile (/= attempted) (entries run) ++ [other]) (transcripts run))

repeated :: PropertyT IO ()
repeated = do
    run <- serial
    expected <- evalEither (replayOf run)
    let journal = entries run
    forM_ [splitAt index journal | (index, Entry.Happened claimed) <- zip [0 ..] journal, repeatable claimed] $ \(leading, rest) -> do
        annotateShow (take 1 rest)
        result <- evalEither (replayOf run {entries = leading ++ take 1 rest ++ rest})
        Core.committed (Replay.state result) === Core.committed (Replay.state expected)
        Replay.statuses result === Replay.statuses expected
        Replay.floors result === Replay.floors expected
        map Replay.confirmed (Map.elems (Replay.learned result)) === map Replay.confirmed (Map.elems (Replay.learned expected))
  where
    repeatable claimed = case claimed of
        Entry.Connected {} -> False
        Entry.Lost {} -> False
        Entry.Started {} -> False
        _ -> True

misplaced :: PropertyT IO ()
misplaced = do
    run <- serial
    let changed rewrite = replayOf run {entries = map rewrite (entries run)}
        reservation replacement recorded = if recorded == Entry.Reserved 0 Entry.Inference 0 0 then replacement else recorded
        attempted replacement recorded = case recorded of
            Entry.Attempted update tried _ number -> Entry.Attempted update tried replacement number
            _ -> recorded
    first' <- firstOf [recorded | recorded@Entry.Dispatched {} <- entries run]
    forM_ [Entry.Reserved 0 Entry.Learner 0 0, Entry.Reserved 0 Entry.Inference 1 0, Entry.Reserved 0 Entry.Inference 0 1] $ \replacement ->
        refused "another role, slot or epoch" (changed (reservation replacement))
    refused "has ended" (replayOf run {entries = concatMap (\recorded -> if recorded == first' then [Entry.Finished 0 (Transcript.Stopped Transcript.Cut), recorded] else [recorded]) (entries run)})
    refused "identity is reused" (changed (attempted (V.ordinal 3)))
    refused "identity is reused" (changed (attempted (V.Binding (V.CallId 9) (V.AttemptId 9) (V.Instance 2))))
    refused "another role or one that has ended" (changed (\recorded -> case recorded of Entry.Attempted update tried bound _ -> Entry.Attempted update tried bound 0; _ -> recorded))

altered :: PropertyT IO ()
altered = do
    run <- serial
    let replace old new = map (\recorded -> if recorded == old then new else recorded)
    storedFirst <- firstOf [recorded | recorded@Entry.Stored {} <- entries run]
    appliedFirst <- firstOf [recorded | recorded@(Entry.Happened Entry.Applied {}) <- entries run]
    completedLast <- firstOf (reverse [recorded | recorded@(Entry.Happened Entry.Completed {}) <- entries run])
    (lastLine, earlier) <- case reverse (Bytes.lines (transcripts run Map.! 0)) of
        final : rest -> pure (final, reverse rest)
        [] -> failure
    record <- evalEither (eitherDecodeStrict lastLine :: Either String Value)
    let changedLine = Lazy.toStrict (encode (Calls.change "extra" (toJSON (0 :: Int)) record))
    refused "has no admitted trajectory" (replayOf run {transcripts = Map.insert 0 (Bytes.unlines (earlier ++ [changedLine])) (transcripts run)})
    case storedFirst of
        Entry.Stored request bound _ -> refused "differs from its admitted trajectory" (replayOf run {entries = replace storedFirst (Entry.Stored request bound (replicate 64 '0')) (entries run)})
        _ -> failure
    case appliedFirst of
        Entry.Happened (Entry.Applied update tried bound identity index consumed before _) -> refused "completion differs from the learner's" (replayOf run {entries = replace appliedFirst (Entry.Happened (Entry.Applied update tried bound identity index consumed before (replicate 64 '0'))) (entries run)})
        _ -> failure
    refused "has no stored result" (replayOf run {entries = filter (/= storedFirst) (entries run)})
    let uncompleted = takeWhile (/= completedLast) (entries run)
    result <- evalEither (replayOf run {entries = uncompleted})
    case completedLast of
        Entry.Happened (Entry.Completed _ _ request digest) -> do
            [status | (chosen, status) <- Map.elems (Replay.statuses result), chosen == request] === [Replay.Stored digest]
            length [() | (_, Replay.Completed _) <- Map.elems (Replay.statuses result)] === fromIntegral (members run) - 1
        _ -> failure
    Core.learning (Replay.state result) === Nothing

claims :: PropertyT IO ()
claims = do
    run <- serial
    let other = replicate 64 '0'
        changed rewrite = replayOf run {entries = map rewrite (entries run)}
    refused "readiness differs" (changed (\recorded -> case recorded of Entry.Happened (Entry.Ready update tried bound identity _) -> Entry.Happened (Entry.Ready update tried bound identity other); _ -> recorded))
    refused "readiness differs" (changed (\recorded -> case recorded of Entry.Happened (Entry.Ready update tried bound _ before) -> Entry.Happened (Entry.Ready update tried bound other before); _ -> recorded))
    refused "step differs" (changed (\recorded -> case recorded of Entry.Happened (Entry.Current update tried index _) -> Entry.Happened (Entry.Current update tried index other); _ -> recorded))
    refused "staged adapter differs" (changed (\recorded -> case recorded of Entry.Happened (Entry.Staged update tried _) -> Entry.Happened (Entry.Staged update tried other); _ -> recorded))
    refused "verified learner differs" (changed (\recorded -> case recorded of Entry.Verified update tried _ -> Entry.Verified update tried other; _ -> recorded))

exits :: PropertyT IO ()
exits = do
    run <- serial
    let session = Entry.Finished 0 (Transcript.Exited ExitSuccess Transcript.Complete)
        replaced outcome = map (\recorded -> if recorded == session then Entry.Finished 0 outcome else recorded) (entries run)
    assert (session `elem` entries run)
    forM_ [Transcript.Exited (ExitFailure 1) Transcript.Complete, Transcript.Exited ExitSuccess Transcript.Cut, Transcript.Stopped Transcript.Complete] $ \outcome ->
        refused "has no admitted trajectory" (replayOf run {entries = replaced outcome})
    result <- evalEither (replayOf run {entries = takeWhile (/= session) (entries run)})
    [() | (_, Replay.Unfinished _) <- Map.elems (Replay.statuses result)] === replicate (fromIntegral (members run)) ()

results :: PropertyT IO ()
results = do
    run <- serial
    verified <- firstOf [recorded | recorded@Entry.Verified {} <- entries run]
    applied <- firstOf (reverse [recorded | recorded@(Entry.Happened Entry.Applied {}) <- entries run])
    let learner' = learnerProcess run
        written = F.wire (records run)
        staging name change = F.wire [if F.stage record == Just name then change record else record | record <- records run]
        resulting = staging "result"
        other = object ["call" .= (9 :: Int), "attempt" .= (9 :: Int), "instance" .= (9 :: Int)]
        changes =
            [ written <> "not a record\n"
            , written <> "{\"stage\""
            , resulting (Calls.change "binding" other)
            , resulting (\record -> Calls.change "request" (Calls.change "extra" (toJSON (0 :: Int)) (Calls.field "request" record)) record)
            , staging "reward_update" (Calls.change "cpu_seconds" (toJSON (-1 :: Double)))
            ]
    forM_ changes $ \learning -> do
        let transcribed = Map.insert learner' learning (transcripts run)
        refused "has no admitted result" (Replay.replay (declaration run) (through verified (entries run)) transcribed)
        _ <- evalEither (Replay.replay (declaration run) (through applied (entries run)) transcribed)
        pure ()

boundaries :: PropertyT IO ()
boundaries = do
    run <- serial
    let exited = Entry.Finished (learnerProcess run) (Transcript.Exited ExitSuccess Transcript.Complete)
        replaced outcome = map (\recorded -> if recorded == exited then Entry.Finished (learnerProcess run) outcome else recorded) (entries run)
    refused "before its process exited cleanly" (replayOf run {entries = filter (/= exited) (entries run)})
    forM_ [Transcript.Exited (ExitFailure 1) Transcript.Complete, Transcript.Stopped Transcript.Complete] $ \outcome ->
        refused "before its process exited cleanly" (replayOf run {entries = replaced outcome})
    owning <- built R.Serial Learner.Resident
    applied <- firstOf (reverse [recorded | recorded@(Entry.Happened Entry.Applied {}) <- entries owning])
    let unreleased = Map.insert (learnerProcess owning) (F.wire [record | record <- records owning, F.stage record /= Just "released"]) (transcripts owning)
    verified <- firstOf [recorded | recorded@Entry.Verified {} <- entries owning]
    refused "result was never released" (Replay.replay (declaration owning) (through verified (entries owning)) unreleased)
    _ <- evalEither (Replay.replay (declaration owning) (through applied (entries owning)) unreleased)
    pure ()

redispatched :: PropertyT IO ()
redispatched = do
    run <- serial
    started <- firstOf [recorded | recorded@(Entry.Happened Entry.Started {}) <- entries run]
    completed <- firstOf [recorded | recorded@(Entry.Happened Entry.Completed {}) <- entries run]
    let recording = through (Entry.Happened (Entry.Recorded 0 0)) (entries run)
        reconnected = [Entry.Restarted [], Entry.Opened 0 1, Entry.Happened (Entry.Connected 0 1)]
    case (started, completed) of
        (Entry.Happened (Entry.Started _ _ request), Entry.Happened (Entry.Completed _ _ finishedRequest digest)) -> do
            refused "without a call dispatched since the last restart" (Replay.replay (declaration run) (recording ++ reconnected ++ [Entry.Happened (Entry.Started 0 1 request)]) (transcripts run))
            refused "without a call dispatched since the last restart" (Replay.replay (declaration run) (recording ++ reconnected ++ [Entry.Happened (Entry.Completed 0 1 finishedRequest digest)]) (transcripts run))
            refused "another worker or epoch" (replayOf run {entries = map (\recorded -> if recorded == started then Entry.Happened (Entry.Started 1 0 request) else recorded) (entries run)})
        _ -> failure

drained :: PropertyT IO ()
drained = do
    run <- built R.Resident Learner.Process
    let isStored Entry.Stored {} = True
        isStored _ = False
        closed = takeWhile (not . isStored) (entries run) ++ [Entry.Finished 0 (Transcript.Exited ExitSuccess Transcript.Complete)]
    result <- evalEither (replayOf run {entries = closed})
    length [() | (_, Replay.Admitted _) <- Map.elems (Replay.statuses result)] === fromIntegral (members run)

cut :: PropertyT IO ()
cut = do
    run <- serial
    let applying = [recorded | recorded@(Entry.Happened Entry.Applied {}) <- entries run]
        shortened = takeWhile ((/= Just "reward_update") . F.stage) (records run)
        unapplied = takeWhile ((/= Just "applied") . F.stage) (records run)
        partial = Bytes.take (Bytes.length (F.wire shortened) + 7) (F.wire (records run))
    current <- firstOf [recorded | recorded@(Entry.Happened Entry.Current {}) <- entries run]
    applied <- firstOf applying
    forM_ [F.wire shortened, partial] $ \learning -> do
        result <- evalEither (Replay.replay (declaration run) (through current (entries run)) (Map.insert 1 learning (transcripts run)))
        case Map.lookup (0, 0) (Replay.learned result) of
            Just learned -> do
                Replay.confirmed learned === 0
                length (Replay.completions learned) === length applying
                case Replay.outcome learned of
                    Replay.Incomplete _ -> success
                    other -> annotateShow other >> failure
            Nothing -> failure
        case (Core.learning (Replay.state result), current) of
            (Just (Update 0, phase), Entry.Happened (Entry.Current _ _ index before)) -> phase === Answering (Attempt 0) index before
            _ -> failure
    refused "was not reported by the learner" (Replay.replay (declaration run) (through applied (entries run)) (Map.insert 1 (F.wire unapplied) (transcripts run)))

outcomes :: PropertyT IO ()
outcomes = do
    run <- serial
    ready <- firstOf [recorded | recorded@(Entry.Happened Entry.Ready {}) <- entries run]
    let recording = through (Entry.Happened (Entry.Recorded 0 0)) (entries run)
        replayedWith journal = evalEither (Replay.replay (declaration run) journal (transcripts run))
        outcomeOf result = Replay.outcome <$> Map.lookup (0, 0) (Replay.learned result)
    concluded <- replayedWith recording
    outcomeOf concluded === Just Replay.Concluded
    Map.null (Replay.evidence concluded) === True
    confirmed <- replayedWith (recording ++ [Entry.Restarted [published run]])
    outcomeOf confirmed === Just (Replay.Committed 1)
    fmap Replay.committing (Map.lookup 1 (Replay.evidence confirmed)) === Just (0, 0)
    redone <- replayedWith (recording ++ [Entry.Restarted []])
    outcomeOf redone === Just Replay.Concluded
    started <- replayedWith (through ready (entries run))
    case outcomeOf started of
        Just (Replay.Incomplete _) -> success
        other -> annotateShow other >> failure

generations :: PropertyT IO ()
generations = do
    run <- serial
    result <- evalEither (replayOf run)
    evidenced <- evalMaybe (Map.lookup 1 (Replay.evidence result))
    let settings = Loop.settings (Replay.config (declaration run))
        built' cohort = Generation.generation settings ("output", "rename") (tasks run, cohort) (Replay.execution evidenced) ([], [], [])
        stepping = [Lazy.toStrict (encode record) | record <- records run, F.stage record `elem` map Just ["proximal", "reference", "current"]]
    selected <- evalEither (built' (Replay.trajectories evidenced))
    Generation.publication selected === Generation.Publication ("output" </> "generation1") "rename"
    assert (not (null stepping))
    Generation.stepOutputs selected === stepping
    assert (isLeft (built' (drop 1 (Replay.trajectories evidenced))))
    assert (isLeft (Generation.generation settings ("output", "copy") (tasks run, Replay.trajectories evidenced) (Replay.execution evidenced) ([], [], [])))

restarts :: PropertyT IO ()
restarts = do
    run <- serial
    initial <- described
    let recording = through (Entry.Happened (Entry.Recorded 0 0)) (entries run)
        staging = through (Entry.Happened (Entry.Staged 0 0 (Entry.adapter (published run)))) (entries run)
        restarted journal observed = Replay.replay (declaration run) (journal ++ [Entry.Restarted observed]) (transcripts run)
        generation = published run
    confirmed <- evalEither (restarted recording [generation])
    Core.committed (Replay.state confirmed) === [Update 0]
    Map.lookup 1 (Replay.versions confirmed) === Just (Entry.adapter generation, Entry.learner generation, Entry.description generation)
    redone <- evalEither (restarted recording [])
    Core.committed (Replay.state redone) === []
    Map.member 1 (Replay.versions redone) === False
    Replay.floors redone === Replay.Floors (members run + 1) 2 1 1
    _ <- evalEither (restarted (entries run) [generation])
    refused "differs from the attempt that recorded it" (restarted recording [generation {Entry.learner = replicate 64 'c'}])
    refused "differs from the attempt that recorded it" (restarted recording [generation {Entry.description = initial}])
    refused "is committed in the journal but its generation is missing" (restarted (entries run) [])
    refused "differs from the update its attempt staged" (restarted (entries run) [generation {Entry.adapter = replicate 64 'c'}])
    refused "without a recorded attempt" (restarted staging [generation])
    refused "not consecutive" (restarted recording [generation {Entry.version = 2}])
    reopened <- firstOf [recorded | recorded@Entry.Dispatched {} <- entries run]
    refused "has ended" (Replay.replay (declaration run) (recording ++ [Entry.Restarted [], reopened]) (transcripts run))
    current <- firstOf [recorded | recorded@(Entry.Happened Entry.Current {}) <- entries run]
    let interrupted = through current (entries run) ++ [Entry.Restarted [], Entry.Attempted 0 1 (V.ordinal (members run + 1)) (learnerProcess run)]
    refused "another role or one that has ended" (Replay.replay (declaration run) interrupted (transcripts run))
