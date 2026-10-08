{-# LANGUAGE OverloadedStrings #-}

module Invar.Async.Replay (Declaration (..), Status (..), Outcome (..), Learned (..), Evidence, Floors (..), Replayed, replay, resume, state, versions, statuses, learned, evidence, floors, reserved, unended, loaded, committing, trajectories, execution, report, frames, admittedUnder) where

import Control.Monad (foldM, forM_, unless, when, zipWithM)
import Data.Aeson (Object, Value (..), withObject)
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.List (sortOn)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe)
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8)
import Invar.Async.Completion (Completion)
import Invar.Async.Completion qualified as Completion
import Invar.Async.Core (Attempt (..), Epoch (..), Worker (..))
import Invar.Async.Core qualified as Core
import Invar.Async.Entry qualified as Entry
import Invar.Async.Plan (Declared (..), Request (..), Update (..), Version (..))
import Invar.Async.Plan qualified as Plan
import Invar.Cohort qualified as C
import Invar.Infer.Batch qualified as Batch
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Session qualified as Session
import Invar.Infer.Trajectory (Trajectory)
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Json qualified as Json
import Invar.Learn qualified as L
import Invar.Learn.Framing qualified as Learner
import Invar.Learn.Report qualified as Report
import Invar.Learn.Stream qualified as S
import Invar.Learn.Trace qualified as Trace
import Invar.Learn.Worker qualified as W
import Invar.Loop qualified as Loop
import Invar.Policy qualified as Policy
import Invar.Resident qualified as Boundary
import Invar.Resident.Owner qualified as Owner
import Invar.Rollout qualified as R
import Invar.Spec.Invocation qualified as V
import Invar.Transcript qualified as Transcript
import Numeric.Natural (Natural)
import System.Exit (ExitCode (..))

data Declaration = Declaration
    { config :: Loop.Config
    , staleness :: Natural
    , plan :: Plan.Plan
    , cycles :: [String -> Either String Loop.Cycle]
    , offsets :: [Natural]
    , initial :: Policy.Description
    }

data Status = Unfinished String | Admitted String | Stored String | Completed String
    deriving (Eq, Show)

data Outcome = Committed Natural | Concluded | Incomplete String
    deriving (Eq, Show)

data Learned = Learned {binding :: V.Binding, process :: Natural, consumed :: [V.Binding], completions :: [Completion], confirmed :: Natural, outcome :: Outcome}

data Evidence = Evidence (Natural, Natural) (L.Settings, [C.Task]) [Replay.Logged] Trace.Attempt Report.Report

committing :: Evidence -> (Natural, Natural)
committing (Evidence key _ _ _ _) = key

admittedUnder :: Evidence -> (L.Settings, [C.Task])
admittedUnder (Evidence _ declaration _ _ _) = declaration

trajectories :: Evidence -> [Replay.Logged]
trajectories (Evidence _ _ logged _ _) = logged

execution :: Evidence -> Trace.Attempt
execution (Evidence _ _ _ attempted _) = attempted

report :: Evidence -> Report.Report
report (Evidence _ _ _ _ reported) = reported

frames :: Evidence -> [Framing.Frame]
frames = Trace.records . execution

data Floors = Floors {identity :: Natural, numbered :: Natural, epoch :: Natural, attempt :: Natural}
    deriving (Eq, Show)

data Replayed = Replayed {state :: Core.State, versions :: Map Natural (String, String, Policy.Description), statuses :: Map V.Binding (Natural, Status), learned :: Map (Natural, Natural) Learned, evidence :: Map Natural Evidence, reserved :: [Natural], unended :: [Natural], loaded :: Map Natural (Entry.Role, [Object]), floors :: Floors}

data Life = Running | Ended (Maybe Transcript.Outcome)
    deriving (Eq)

data Process = Process Entry.Role Natural Natural Life

data Tail = Whole | Fragment | Unreadable String

data Inference = Finite [V.Binding] | Hosting (Either String ((Session.Session, Owner.State), [Framing.Frame])) Tail [V.Binding]

data Learner = Launched | Owning (Either String (Owner.State, [Framing.Frame])) Tail

data Boundary = Exit | Release (Either String ())

data Tried = Tried {tryBinding :: V.Binding, tryProcess :: Natural, inputs :: [V.Binding], declaredFor :: Maybe (L.Settings, [C.Task]), traced :: Either String Trace.Attempt, boundary :: Boundary, applied :: Natural, staged :: Maybe String, verified :: Maybe String}

data Generation = Generation String String Policy.Description

data Fold = Fold
    { core :: Core.State
    , processes :: Map Natural Process
    , calls :: Map V.Binding (Natural, Natural, Natural)
    , latest :: Map Natural V.Binding
    , products :: Map V.Binding (Either String Replay.Logged)
    , stored :: Map V.Binding String
    , completed :: Set V.Binding
    , inferences :: Map Natural Inference
    , learners :: Map Natural Learner
    , tries :: Map (Natural, Natural) Tried
    , generations :: Map Natural Generation
    , committers :: Map Natural (Natural, Natural)
    , identities :: Set V.Binding
    , epochs :: Natural
    , prefixes :: Map Natural [Framing.Frame]
    }

replay :: Declaration -> [Entry.Entry] -> Map Natural ByteString -> Either String Replayed
replay declared entries transcripts = folded declared entries transcripts >>= drained declared

resume :: Declaration -> [Entry.Entry] -> Map Natural ByteString -> [Entry.Generation] -> Either String (Replayed, [Core.Command])
resume declared entries transcripts observed = do
    (restartedFold, commands) <- folded declared entries transcripts >>= (`restarted` observed)
    replayed <- drained declared restartedFold
    pure (replayed, commands)

folded :: Declaration -> [Entry.Entry] -> Map Natural ByteString -> Either String Fold
folded declared entries transcripts = foldM (entry declared transcripts) started entries
  where
    settings = Loop.settings (config declared)
    started = Fold (fst (Core.start (plan declared))) Map.empty Map.empty Map.empty Map.empty Map.empty Set.empty Map.empty Map.empty Map.empty (Map.singleton 0 (Generation (L.policy settings) (L.learner settings) (initial declared))) Map.empty Set.empty 0 Map.empty

drained :: Declaration -> Fold -> Either String Replayed
drained declared fold = summarize <$> foldM (settle declared) fold (Map.keys (inferences fold))

entry :: Declaration -> Map Natural ByteString -> Fold -> Entry.Entry -> Either String Fold
entry declared transcripts fold recorded = case recorded of
    Entry.Declared _ _ -> Left "The journal declares its run twice"
    Entry.Opened _ used -> pure fold {epochs = max (epochs fold) (used + 1)}
    Entry.Reserved number chosen place used -> do
        when (Map.member number (processes fold)) (Left "A process number is reserved twice")
        unless (chosen `elem` [inferenceRole declared, learnerRole declared]) (Left "A process is reserved under a role the declared modes do not run")
        when (chosen == Entry.Shared && or [life == Running | Process Entry.Shared _ _ life <- Map.elems (processes fold)]) (Left "A shared process is reserved while another shared process runs")
        pure fold {processes = Map.insert number (Process chosen place used Running) (processes fold), epochs = max (epochs fold) (used + 1)}
    Entry.Dispatched request worker used bound number -> do
        running fold number (inferenceRole declared, worker, used)
        encoded <- transcript transcripts number
        fresh fold bound
        update <- maybe (Left "A dispatched request belongs to no update") Right (Plan.owner (plan declared) (Request request))
        let Version selected = Plan.version (plan declared) update
            replayed = case fromMaybe (hostedOwner declared worker encoded) (Map.lookup number (inferences fold)) of
                Finite pending -> Finite (pending ++ [bound])
                Hosting state ending pending -> Hosting state ending (pending ++ [bound])
        pure fold {calls = Map.insert bound (request, number, selected) (calls fold), latest = Map.insert request bound (latest fold), inferences = Map.insert number replayed (inferences fold), identities = Set.insert bound (identities fold)}
    Entry.Attempted update tried bound number -> do
        Process chosen place _ life <- maybe (Left "An attempt names a process that was never reserved") Right (Map.lookup number (processes fold))
        unless (chosen == learnerRole declared && life == Running) (Left "An attempt names a process of another role or one that has ended")
        unless (Core.learning (core fold) == Just (Update update, Core.Loading (Attempt tried))) (Left "An attempt is journaled under a number the core did not assign")
        encoded <- transcript transcripts number
        fresh fold bound
        when (Map.member (update, tried) (tries fold)) (Left "An attempt is journaled twice")
        let declaredInput = learnerInput declared fold update bound
            used = either (const []) (\(_, _, bindings, _) -> bindings) declaredInput
            under = either (const Nothing) (\(settings, _, _, tasks) -> Just (settings, tasks)) declaredInput
            input' = (\(settings, invoked, _, _) -> (settings, invoked)) <$> declaredInput
        (placed, attempted, release, loading) <- case (Loop.learningMode (Loop.backend (config declared)), Map.lookup number (learners fold)) of
            (W.Shared, _) -> do
                unless (all (\consumed -> fmap (\(_, owner, _) -> owner) (Map.lookup consumed (calls fold)) == Just number) used) (Left "A shared learner attempt names another process than the one that ran its calls")
                case Map.lookup number (inferences fold) of
                    Just (Hosting state ending _) -> do
                        let (next, attempted', release', loading') = residentAttempt input' (fmap (\((_, physical), remaining) -> (physical, remaining)) state)
                            cursor = (\((session, _), _) (physical, remaining) -> ((session, physical), remaining)) <$> state <*> next
                        pure (fold {inferences = Map.insert number (Hosting cursor ending []) (inferences fold)}, attempted', release', loading')
                    _ -> Left "A shared learner attempt names a process that hosted no inference"
            (W.Resident, owner) -> case fromMaybe (learnerOwner place encoded) owner of
                Owning state ending -> do
                    let (next, attempted', release', loading') = residentAttempt input' state
                    pure (fold {learners = Map.insert number (Owning next ending) (learners fold)}, attempted', release', loading')
                Launched -> Left "A learner process runs a second attempt"
            (_, Nothing) -> do
                let begun = launched input' (output encoded)
                pure (fold {learners = Map.insert number Launched (learners fold)}, snd <$> begun, Exit, either (const []) fst begun)
            _ -> Left "A learner process runs a second attempt"
        pure placed {tries = Map.insert (update, tried) (Tried bound number used under attempted release 0 Nothing Nothing) (tries placed), identities = Set.insert bound (identities placed), prefixes = admittedLoad number loading (prefixes placed)}
    Entry.Finished number outcome -> do
        Process chosen place used life <- maybe (Left "An exit names a process that was never reserved") Right (Map.lookup number (processes fold))
        unless (life == Running) (Left "A process exits twice")
        closed <- finished declared transcripts fold (number, chosen, place) outcome
        pure closed {processes = Map.insert number (Process chosen place used (Ended (Just outcome))) (processes closed)}
    Entry.Stored request bound digest -> do
        (owned, _, _) <- maybe (Left "A stored result names a call that was never dispatched") Right (Map.lookup bound (calls fold))
        unless (owned == request) (Left "A stored result names another request's call")
        (advanced, found) <- produced declared fold bound
        case found of
            Left problem -> Left ("A stored result's call has no admitted trajectory: " ++ problem)
            Right logged
                | Trajectory.digest (Replay.trajectory logged) /= digest -> Left "A stored result differs from its admitted trajectory"
                | otherwise -> pure advanced {stored = Map.insert bound digest (stored advanced)}
    Entry.Happened claimed -> happened fold claimed
    Entry.Verified update tried digest -> do
        finished' <- concluded fold (update, tried)
        reported <- Report.artifact "learner" finished'
        unless (reported == digest) (Left "A verified learner differs from the learner's result")
        pure fold {tries = Map.adjust (\chosen -> chosen {verified = Just digest}) (update, tried) (tries fold)}
    Entry.Restarted observed -> fst <$> restarted fold observed
    Entry.Elapsed {} -> pure fold

running :: Fold -> Natural -> (Entry.Role, Natural, Natural) -> Either String ()
running fold number (expected, worker, used) = case Map.lookup number (processes fold) of
    Just (Process chosen place reserved life)
        | chosen /= expected || place /= worker || reserved /= used -> Left "A call names a process of another role, slot or epoch"
        | life /= Running -> Left "A call names a process that has ended"
        | otherwise -> pure ()
    Nothing -> Left "A call names a process that was never reserved"

transcript :: Map Natural ByteString -> Natural -> Either String ByteString
transcript transcripts number = maybe (Left "An entry names a process whose transcript is missing") Right (Map.lookup number transcripts)

inferenceOwner :: Boundary.Role -> Natural -> ByteString -> Inference
inferenceOwner role place encoded = let (records, ending) = output encoded in Hosting (Right ((Session.start Session.Resident, Owner.start (Boundary.Owner role place)), records)) ending []

hostedOwner :: Declaration -> Natural -> ByteString -> Inference
hostedOwner declared place encoded = case Loop.inferenceMode (Loop.backend (config declared)) of
    R.Resident -> inferenceOwner Boundary.Inference place encoded
    R.Shared -> inferenceOwner Boundary.Shared place encoded
    _ -> Finite []

inferenceRole :: Declaration -> Entry.Role
inferenceRole declared = if Loop.inferenceMode (Loop.backend (config declared)) == R.Shared then Entry.Shared else Entry.Inference

learnerRole :: Declaration -> Entry.Role
learnerRole declared = if Loop.learningMode (Loop.backend (config declared)) == W.Shared then Entry.Shared else Entry.Learner

learnerOwner :: Natural -> ByteString -> Learner
learnerOwner place encoded = let (records, ending) = output encoded in Owning (Right (Owner.start (Boundary.Owner Boundary.Learning place), records)) ending

fresh :: Fold -> V.Binding -> Either String ()
fresh fold bound = when (any (overlaps bound) (Set.toList (identities fold))) (Left "A call identity is reused")
  where
    overlaps (V.Binding call tried instanceId) (V.Binding otherCall otherTry otherInstance) = call == otherCall || tried == otherTry || instanceId == otherInstance

output :: ByteString -> ([Framing.Frame], Tail)
output encoded = case reverse (Bytes.split '\n' encoded) of
    remainder : complete -> parsed (reverse complete) remainder
    [] -> ([], Whole)
  where
    parsed (line : rest) remainder = case Json.decode line >>= parseEither (withObject "transcript record" pure) of
        Right fields -> let (later, ending) = parsed rest remainder in (Framing.Frame line fields : later, ending)
        Left problem -> ([], Unreadable ("A transcript line is not a record: " ++ problem))
    parsed [] remainder = ([], if Bytes.null remainder then Whole else Fragment)

finished :: Declaration -> Map Natural ByteString -> Fold -> (Natural, Entry.Role, Natural) -> Transcript.Outcome -> Either String Fold
finished declared transcripts prior (number, role, place) outcome = do
    encoded <- transcript transcripts number
    let backend = Loop.backend (config declared)
        fold = case role of
            Entry.Inference | Loop.inferenceMode backend == R.Resident -> prior {inferences = Map.insertWith (\_ held -> held) number (inferenceOwner Boundary.Inference place encoded) (inferences prior)}
            Entry.Shared -> prior {inferences = Map.insertWith (\_ held -> held) number (inferenceOwner Boundary.Shared place encoded) (inferences prior)}
            Entry.Learner | Loop.learningMode backend == W.Resident -> prior {learners = Map.insertWith (\_ held -> held) number (learnerOwner place encoded) (learners prior)}
            _ -> prior
    case (Map.lookup number (inferences fold), Map.lookup number (learners fold)) of
        (Just (Finite bound), _) -> do
            declaration <- declarationOf declared fold bound
            let protocol = if Loop.inferenceMode backend == R.Batched then Session.Batched else Session.Serial
                admitted = Replay.ended protocol declaration outcome encoded
                loading = either (const []) (const (preparation (fst (output encoded)))) admitted
            pure fold {products = Map.union (Map.fromList (attach bound admitted)) (products fold), prefixes = admittedLoad number loading (prefixes fold)}
        (Just Hosting {}, _) | clean -> do
            settled <- settle declared fold number
            case Map.lookup number (inferences settled) of
                Just (Hosting (Right ((current, physical), remaining)) ending []) -> closing (Just current) physical remaining ending >> pure settled
                Just (Hosting (Left problem) _ _) -> Left ("A resident owner exits cleanly after records that cannot be admitted: " ++ problem)
                _ -> Left "A resident owner exits cleanly with calls it never released"
        (_, Just (Owning (Right (physical, remaining)) ending)) | clean -> closing Nothing physical remaining ending >> pure fold
        (_, Just (Owning (Left problem) _)) | clean -> Left ("A resident owner exits cleanly after records that cannot be admitted: " ++ problem)
        _ -> pure fold
  where
    clean = outcome == Transcript.Exited ExitSuccess Transcript.Complete
    closing session physical remaining ending = case (remaining, ending) of
        ([closed], Whole) -> do
            unless (all Session.settled session) (Left "A resident owner closes with active invocation loads")
            Owner.close physical (Framing.raw closed)
        _ -> Left "A resident owner exits without exactly its close acknowledgement"

attach :: [V.Binding] -> Either Session.Error [Replay.Logged] -> [(V.Binding, Either String Replay.Logged)]
attach bound admitted = case admitted of
    Right logged -> [(chosen, maybe (Left "The session admitted no trajectory for the call") Right (lookup chosen [(Trajectory.binding (Replay.trajectory one), one) | one <- logged])) | chosen <- bound]
    Left problem -> [(chosen, Left (show problem)) | chosen <- bound]

produced :: Declaration -> Fold -> V.Binding -> Either String (Fold, Either String Replay.Logged)
produced declared fold bound = case Map.lookup bound (products fold) of
    Just found -> pure (fold, found)
    Nothing -> case Map.lookup bound (calls fold) >>= \(_, number, _) -> (,) number <$> Map.lookup number (inferences fold) of
        Just (number, Hosting state ending pending@(first' : _)) | bound `elem` pending -> do
            let owner request = Plan.owner (plan declared) (Request request)
                requestOf chosen = (\(request, _, _) -> request) <$> Map.lookup chosen (calls fold)
                same chosen = (requestOf chosen >>= owner) == (requestOf first' >>= owner)
                (grouped, later) = span same pending
                failed problem = fold {inferences = Map.insert number (Hosting (Left problem) ending []) (inferences fold), products = Map.union (Map.fromList [(chosen, Left problem) | chosen <- pending]) (products fold)}
            advanced <- case state of
                Left problem -> pure (failed problem)
                Right (hosted, remaining) -> do
                    declaration <- declarationOf declared fold grouped
                    pure $ case Replay.group hosted declaration remaining of
                        Right (next, logged, consumed, rest) -> fold {inferences = Map.insert number (Hosting (Right (next, rest)) ending later) (inferences fold), products = Map.union (Map.fromList (attach grouped (Right logged))) (products fold), prefixes = admittedLoad number (preparation consumed) (prefixes fold)}
                        Left problem -> failed (show problem)
            produced declared advanced bound
        _ -> pure (fold, Left "The call's process never established a result")

declarationOf :: Declaration -> Fold -> [V.Binding] -> Either String Session.Declaration
declarationOf declared fold bound = do
    located <- traverse (\chosen -> maybe (Left "A call was never dispatched") (Right . (,) chosen) (Map.lookup chosen (calls fold))) bound
    chosenVersions <- case Set.toList (Set.fromList [selected | (_, (_, _, selected)) <- located]) of
        [selected] -> pure selected
        _ -> Left "One process session mixes calls of several versions"
    prepared <- traverse (\(chosen, (request, _, selected)) -> taskOf declared fold request selected >>= first show . Call.prepare chosen . C.plan) located
    scoring <- referenceOf declared fold chosenVersions
    pure (Session.Declaration prepared scoring)

taskOf :: Declaration -> Fold -> Natural -> Natural -> Either String C.Task
taskOf declared fold request selected = do
    Update update <- maybe (Left "A request belongs to no update") Right (Plan.owner (plan declared) (Request request))
    tasks <- tasksOf declared fold update selected
    case drop (fromIntegral (request - (offsets declared !! fromIntegral update))) tasks of
        chosen : _ -> pure chosen
        [] -> Left "A request lies outside its declared cycle"

tasksOf :: Declaration -> Fold -> Natural -> Natural -> Either String [C.Task]
tasksOf declared fold update selected = do
    Generation chosen _ described <- generationOf fold selected
    cycle' <- (cycles declared !! fromIntegral update) chosen
    first show (Loop.bindTasks described (Loop.tasks cycle'))

referenceOf :: Declaration -> Fold -> Natural -> Either String (Maybe String)
referenceOf declared fold selected = do
    Generation chosen learnerDigest _ <- generationOf fold selected
    pure (Batch.identity <$> Loop.scoring (config declared) (Loop.Checkpoint "" chosen learnerDigest))

generationOf :: Fold -> Natural -> Either String Generation
generationOf fold selected = maybe (Left "A call uses a version that was never published") Right (Map.lookup selected (generations fold))

happened :: Fold -> Entry.Claim -> Either String Fold
happened fold claimed = case claimed of
    Entry.Connected worker used -> stepped fold (Core.Connected (Worker worker) (Epoch used))
    Entry.Lost worker used -> stepped fold (Core.Lost (Worker worker) (Epoch used))
    Entry.Started worker used request -> do
        _ <- callOf fold (worker, used) request
        stepped fold (Core.Started (Worker worker) (Epoch used) (Request request))
    Entry.Completed worker used request digest -> do
        bound <- callOf fold (worker, used) request
        unless (Map.lookup bound (stored fold) == Just digest) (Left "A completed request has no stored result with its digest")
        advanced <- stepped fold (Core.Completed (Worker worker) (Epoch used) (Request request) digest)
        pure advanced {completed = Set.insert bound (completed advanced)}
    Entry.Ready update tried bound identityDigest before -> do
        record <- tryOf fold (update, tried)
        traced' <- traced record
        let stream = Trace.stream traced'
        unless (tryBinding record == bound && S.exchange stream == bound && S.identity stream == identityDigest && Trace.opening traced' == before) (Left "A journaled readiness differs from the learner's")
        stepped fold (Core.Ready (Update update) (Attempt tried) bound identityDigest before)
    Entry.Current update tried index before -> do
        traced' <- tryOf fold (update, tried) >>= traced
        unless (any (\(reported, _, _) -> reported == index) (S.currents (Trace.stream traced')) && opened traced' index == Just before) (Left "A journaled step differs from the learner's")
        stepped fold (Core.Current (Update update) (Attempt tried) index before)
    Entry.Applied update tried _ _ _ _ _ _ -> do
        record <- tryOf fold (update, tried)
        stream <- Trace.stream <$> traced record
        let (confirmed, unconfirmed) = splitAt (fromIntegral (applied record)) (S.completions stream)
            claimOf done = Entry.claim (Core.Applied (Update update) (Attempt tried) done)
        case (filter ((== claimed) . claimOf) confirmed, unconfirmed) of
            (done : _, _) -> stepped fold (Core.Applied (Update update) (Attempt tried) done)
            ([], done : _) -> do
                unless (claimOf done == claimed) (Left "A journaled completion differs from the learner's")
                stepping <- stepped fold (Core.Applied (Update update) (Attempt tried) done)
                pure stepping {tries = Map.adjust (\chosen -> chosen {applied = applied chosen + 1}) (update, tried) (tries stepping)}
            ([], []) -> Left "A journaled completion was not reported by the learner"
    Entry.Staged update tried digest -> do
        finished' <- concluded fold (update, tried)
        reported <- Report.artifact "adapter" finished'
        unless (reported == digest) (Left "A staged adapter differs from the learner's result")
        stepping <- stepped fold (Core.Staged (Update update) (Attempt tried) digest)
        pure stepping {tries = Map.adjust (\chosen -> chosen {staged = Just digest}) (update, tried) (tries stepping)}
    Entry.Recorded update tried -> stepped fold (Core.Recorded (Update update) (Attempt tried))
    Entry.Committed update tried -> do
        advanced <- stepped fold (Core.Committed (Update update) (Attempt tried))
        published <- successorOf advanced (update, tried)
        pure advanced {generations = Map.insert (update + 1) published (generations advanced), committers = Map.insert (update + 1) (update, tried) (committers advanced)}
    Entry.Abandoned update tried -> stepped fold (Core.Abandoned (Update update) (Attempt tried))

stepped :: Fold -> Core.Event -> Either String Fold
stepped fold event = do
    (next, _) <- first show (Core.step (core fold) event)
    pure fold {core = next}

opened :: Trace.Attempt -> Natural -> Maybe String
opened attempted index
    | index == 0 = Just (Trace.opening attempted)
    | otherwise = case drop (fromIntegral index - 1) (S.completions (Trace.stream attempted)) of
        done : _ -> Just (Completion.after done)
        [] -> Nothing

tryOf :: Fold -> (Natural, Natural) -> Either String Tried
tryOf fold key = maybe (Left "A learner event names an attempt that was never journaled") Right (Map.lookup key (tries fold))

successorOf :: Fold -> (Natural, Natural) -> Either String Generation
successorOf fold (update, tried) = do
    record <- tryOf fold (update, tried)
    adapter <- maybe (Left "A committed attempt staged no adapter") Right (staged record)
    learnerDigest <- maybe (Left "A committed attempt verified no learner") Right (verified record)
    Generation _ _ previous <- generationOf fold update
    described <- Policy.successor adapter previous
    pure (Generation adapter learnerDigest described)

callOf :: Fold -> (Natural, Natural) -> Natural -> Either String V.Binding
callOf fold (worker, used) request = do
    bound <- maybe (Left "A request is reported without a call dispatched since the last restart") Right (Map.lookup request (latest fold))
    reporting <- case Map.lookup bound (calls fold) >>= \(_, number, _) -> Map.lookup number (processes fold) of
        Just (Process _ place reserved _) -> pure (place, reserved)
        Nothing -> Left "A request's call names no reserved process"
    unless (reporting == (worker, used)) (Left "A request is reported by another worker or epoch than the process its call was dispatched to")
    pure bound

concluded :: Fold -> (Natural, Natural) -> Either String Report.Report
concluded fold key = do
    record <- tryOf fold key
    attempted <- traced record
    finished' <- maybe (Left ("A learner attempt has no admitted result" ++ maybe "" (": " ++) (Trace.stopped attempted))) Right (Trace.result attempted)
    case boundary record of
        Release accepted -> either (Left . ("A resident learner result was never released: " ++)) pure accepted
        Exit -> case Map.lookup (tryProcess record) (processes fold) of
            Just (Process _ _ _ (Ended (Just (Transcript.Exited ExitSuccess Transcript.Complete)))) -> pure ()
            _ -> Left "A learner result is used before its process exited cleanly"
    pure finished'

learnerInput :: Declaration -> Fold -> Natural -> V.Binding -> Either String (L.Settings, (V.Binding, Text, Value), [V.Binding], [C.Task])
learnerInput declared fold update bound = do
    Generation current currentLearner _ <- generationOf fold update
    let behavior = if update > staleness declared then update - staleness declared else 0
    Generation chosen _ _ <- generationOf fold behavior
    let settings = (Loop.settings (config declared)) {L.policy = current, L.learner = currentLearner, L.schedule = L.Schedule update (staleness declared) behavior chosen}
    Declared requests _ <- maybe (Left "An attempt names an undeclared update") Right (Plan.declared (plan declared) (Update update))
    used <- traverse (\(Request request) -> trajectoryOf fold request) requests
    tasks <- tasksOf declared fold update behavior
    (program, payload) <-
        either
            (Left . show)
            id
            ( C.withCohort
                (C.Definition chosen tasks)
                ( \cohort -> do
                    observations <- first show (zipWithM C.record (C.members cohort) (map snd used))
                    batch <- first show (C.admit cohort observations)
                    first show (L.observedInput settings batch)
                )
            )
    request <- Json.decode payload
    pure (settings, (bound, decodeUtf8 program, request), map fst used, tasks)

trajectoryOf :: Fold -> Natural -> Either String (V.Binding, Trajectory)
trajectoryOf fold request = do
    bound <- maybe (Left "An update's request was never dispatched") Right (Map.lookup request (latest fold))
    unless (Set.member bound (completed fold)) (Left "An update's request has no completed result")
    case Map.lookup bound (products fold) of
        Just (Right logged) -> pure (bound, Replay.trajectory logged)
        _ -> Left "An update's request has no admitted trajectory"

launched :: Either String (L.Settings, (V.Binding, Text, Value)) -> ([Framing.Frame], Tail) -> Either String ([Framing.Frame], Trace.Attempt)
launched declaredInput (records, ending) = do
    (settings, invoked) <- declaredInput
    (loading, attempted) <- Learner.finite settings invoked records
    pure $ (,) loading $ case (ending, Trace.result attempted) of
        (Unreadable problem, _) -> Trace.stop problem attempted
        (Fragment, Just _) -> Trace.stop "Output follows the learner result" attempted
        _ -> attempted

residentAttempt :: Either String (L.Settings, (V.Binding, Text, Value)) -> Either String (Owner.State, [Framing.Frame]) -> (Either String (Owner.State, [Framing.Frame]), Either String Trace.Attempt, Boundary, [Framing.Frame])
residentAttempt declaredInput state = case (declaredInput, state) of
    (Right (settings, invoked), Right (physical, remaining)) -> case Learner.resident settings physical invoked remaining of
        Right (attempted, admitted, Right (Learner.Released next _ after)) -> (Right (next, after), Right attempted, Release (Right ()), preparation admitted)
        Right (attempted, _, Left problem) -> (Left ("An earlier attempt on the owner was never released: " ++ problem), Right attempted, Release (Left problem), [])
        Left problem -> (Left problem, Left problem, Release (Left problem), [])
    (Left problem, _) -> (Left problem, Left problem, Release (Left problem), [])
    (_, Left problem) -> (Left problem, Left problem, Release (Left problem), [])

preparation :: [Framing.Frame] -> [Framing.Frame]
preparation = takeWhile ((`elem` map (Just . String) ["loading", "profile", "load"]) . Framing.stageName)

admittedLoad :: Natural -> [Framing.Frame] -> Map Natural [Framing.Frame] -> Map Natural [Framing.Frame]
admittedLoad number loading current
    | null loading = current
    | otherwise = Map.insertWith (\_ earlier -> earlier) number loading current

restarted :: Fold -> [Entry.Generation] -> Either String (Fold, [Core.Command])
restarted fold observed = do
    let committedCount = length (Core.committed (core fold))
        sorted = sortOn Entry.version observed
        matches seen (Generation chosen learnerDigest described) = Entry.adapter seen == chosen && Entry.learner seen == learnerDigest && Entry.description seen == described
        named seen = "Generation " ++ show (Entry.version seen)
    unless (map Entry.version sorted == [1 .. fromIntegral (length sorted)]) (Left "Observed generations are not consecutive from the first")
    when (length sorted < committedCount) (Left ("Update " ++ show (length sorted) ++ " is committed in the journal but its generation is missing"))
    when (length sorted > committedCount + 1) (Left "More generations are published than the journal committed")
    forM_ (take committedCount sorted) $ \seen -> do
        replayed <- generationOf fold (Entry.version seen)
        unless (matches seen replayed) (Left (named seen ++ " differs from the update its attempt staged"))
    confirmedPublication <- case drop committedCount sorted of
        [] -> pure Nothing
        seen : _ -> case Core.learning (core fold) of
            Just (Update update, Core.Committing (Attempt tried)) | update == fromIntegral committedCount -> do
                successor <- successorOf fold (update, tried)
                unless (matches seen successor) (Left (named seen ++ " differs from the attempt that recorded it"))
                pure (Just (Update update, Attempt tried, successor))
            _ -> Left (named seen ++ " is published without a recorded attempt")
    (next, commands) <- first show (Core.restart (core fold) ((\(update, tried, _) -> (update, tried)) <$> confirmedPublication))
    let (published, committing') = case confirmedPublication of
            Just (Update update, Attempt tried, chosen) -> (Map.insert (update + 1) chosen (generations fold), Map.insert (update + 1) (update, tried) (committers fold))
            Nothing -> (generations fold, committers fold)
    pure (fold {core = next, generations = published, committers = committing', latest = Map.empty, processes = Map.map (\(Process chosen place used life) -> Process chosen place used (if life == Running then Ended Nothing else life)) (processes fold)}, commands)

settle :: Declaration -> Fold -> Natural -> Either String Fold
settle declared fold number = case Map.lookup number (inferences fold) of
    Just (Hosting _ _ (pending : _)) -> do
        (advanced, _) <- produced declared fold pending
        settle declared advanced number
    _ -> pure fold

summarize :: Fold -> Replayed
summarize fold = Replayed (core fold) published classified attempted committed (Map.keys (processes fold)) [number | (number, Process _ _ _ Running) <- Map.toList (processes fold)] (Map.intersectionWith (\(Process chosen _ _ _) loading -> (chosen, map Framing.fields loading)) (processes fold) (prefixes fold)) (Floors next numbers (epochs fold) (Core.issued (core fold)))
  where
    published = Map.map (\(Generation chosen learnerDigest described) -> (chosen, learnerDigest, described)) (generations fold)
    classified = Map.mapWithKey (\bound (request, _, _) -> (request, status bound)) (calls fold)
    status bound
        | Set.member bound (completed fold) = Completed (stored fold Map.! bound)
        | Just digest <- Map.lookup bound (stored fold) = Stored digest
        | Just (Right logged) <- Map.lookup bound (products fold) = Admitted (Trajectory.digest (Replay.trajectory logged))
        | Just (Left problem) <- Map.lookup bound (products fold) = Unfinished problem
        | otherwise = Unfinished "The call's process ended before it was replayed"
    attempted = Map.mapWithKey (\key record -> Learned (tryBinding record) (tryProcess record) (inputs record) (either (const []) (S.completions . Trace.stream) (traced record)) (applied record) (outcomeOf key)) (tries fold)
    versionOf = Map.fromList [(key, version) | (version, key) <- Map.toList (committers fold)]
    outcomeOf key = case (Map.lookup key versionOf, concluded fold key) of
        (Just version, _) -> Committed version
        (Nothing, Right _) -> Concluded
        (Nothing, Left reason) -> Incomplete reason
    committed = Map.mapMaybe evidenced (committers fold)
    evidenced key = do
        record <- Map.lookup key (tries fold)
        finished' <- either (const Nothing) Just (concluded fold key)
        admission <- either (const Nothing) Just (traced record)
        consumed' <- traverse (\bound -> case Map.lookup bound (products fold) of Just (Right logged) -> Just logged; _ -> Nothing) (inputs record)
        declaration <- declaredFor record
        pure (Evidence key declaration consumed' admission finished')
    next = maybe 0 (+ 1) (Set.lookupMax (Set.map largest (identities fold)))
    numbers = maybe 0 ((+ 1) . fst) (Map.lookupMax (processes fold))
    largest (V.Binding (V.CallId call) (V.AttemptId tried) (V.Instance instanceId)) = maximum [call, tried, instanceId]
