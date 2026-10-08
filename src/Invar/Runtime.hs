{-# LANGUAGE OverloadedStrings #-}

module Invar.Runtime (Error (..), Run (..), run, resume, interpret, declaration) where

import Control.Concurrent (forkFinally, killThread)
import Control.Concurrent.Chan (Chan, newChan, readChan, writeChan)
import Control.Concurrent.MVar (MVar, modifyMVar, modifyMVar_, newEmptyMVar, newMVar, putMVar, readMVar, tryPutMVar)
import Control.Exception (SomeAsyncException, bracket, finally, fromException, throwIO, try)
import Control.Monad (forM_, forever, unless, void, when)
import Data.Aeson (Object, Value, encode, object, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString.Char8 qualified as Char
import Data.ByteString.Lazy qualified as Lazy
import Data.IORef (atomicModifyIORef', newIORef, readIORef, writeIORef)
import Data.List (genericLength, genericTake)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import GHC.Clock (getMonotonicTime)
import Invar.Async.Core (Attempt (..), Command (..), Epoch (..), Event (..), Worker (..))
import Invar.Async.Core qualified as Core
import Invar.Async.Entry qualified as Entry
import Invar.Async.Plan (Declared (..), Request (..), Update (..), Version (..))
import Invar.Async.Plan qualified as Plan
import Invar.Async.Recorded qualified as Recorded
import Invar.Async.Replay qualified as Replay
import Invar.Cohort qualified as C
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Journal qualified as Journal
import Invar.Learn qualified as L
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Stream qualified as S
import Invar.Learn.Worker qualified as W
import Invar.Learn.Worker.Observation qualified as Observation
import Invar.Learn.Worker.Owner qualified as Owner
import Invar.Learn.Worker.Resident qualified as Resident
import Invar.Loop qualified as Loop
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as R
import Invar.Rollout.Internal qualified as Internal
import Invar.Spec.Invocation qualified as V
import Invar.Store qualified as Store
import Invar.Transcript qualified as Transcript
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)
import System.Directory (canonicalizePath, createDirectory, getCurrentDirectory, withCurrentDirectory)
import System.FilePath ((</>))
import System.IO.Error (tryIOError)

data Error = Declaration String | Recovery String | Rollout R.Error | Admission L.Error | Learning W.Failure | Unresolved Natural Error | Crashed String
    deriving (Show)

data Run = Run Loop.Config Natural Workload.Document

data Published = Published Loop.Checkpoint Policy.Description

data Origin = Origin Core.State [Command] (Map Natural Published) Natural Natural Natural

data Resumed = Resumed FilePath Run Plan.Plan [Entry.Generation] Origin

data Shared scope = Shared
    { config :: Loop.Config
    , staleness :: Natural
    , plan :: Plan.Plan
    , cycles :: [String -> Either String Loop.Cycle]
    , offsets :: [Natural]
    , recorder :: Recorder
    , core :: MVar Core.State
    , published :: MVar (Map Natural Published)
    , batches :: MVar (Map Natural (MVar (R.Batch scope)))
    , pending :: MVar (Map Natural [Request])
    , cohorts :: Chan (Natural, Natural)
    , updates :: Chan (Natural, Attempt)
    , finished :: MVar (Either Error ())
    , invoked :: MVar (Maybe Natural)
    , driver :: R.Driver scope
    , runner :: Owner.Runner
    }

data Recorder = Recorder
    { journal :: Journal.Journal
    , transcripts :: FilePath
    , numbers :: MVar Natural
    , epoch :: Natural
    , processOf :: MVar (Map Natural Natural)
    , launching :: MVar (Maybe Transcript.Transcript)
    , owning :: MVar (Maybe Natural)
    }

run :: Run -> Value -> IO (Either Error ())
run selected@(Run chosen _ document) arguments = do
    prepared <- prepare selected
    case prepared of
        Left problem -> pure (Left problem)
        Right (planned, initial) -> do
            started <- getCurrentDirectory
            createDirectory (Loop.root chosen)
            createDirectory (Loop.root chosen </> "transcripts")
            Journal.with (Loop.root chosen </> "journal.jsonl") (Entry.encode (Entry.Declared started (Fields.fromList ["arguments" .= arguments, "workload" .= Workload.value document]))) $ \recorded -> do
                let (state, commands) = Core.start planned
                begin selected planned (Origin state commands (Map.singleton 0 initial) 0 0 0) recorded

resume :: FilePath -> (Value -> Either String (Loop.Config, Natural)) -> IO (Either Error ())
resume directory interpreter = do
    target <- canonicalizePath directory
    either Left id <$> Journal.resume (target </> "journal.jsonl") (admit target interpreter) proceed

admit :: FilePath -> (Value -> Either String (Loop.Config, Natural)) -> [Object] -> IO (Either Error Resumed)
admit target interpreter recorded = case traverse (parseEither Entry.decode) recorded of
    Left problem -> pure (Left (Recovery problem))
    Right (Entry.Declared started declared : later) ->
        withCurrentDirectory started $ do
            interpreted <- interpret target interpreter declared
            prepared <- either (pure . Left) (\selected -> fmap (selected,) <$> declaration selected) interpreted
            case prepared of
                Left problem -> pure (Left problem)
                Right (selected@(Run chosen _ _), replaying) -> do
                    found <- tryIOError ((,) <$> Recorded.transcripts (Loop.root chosen) later <*> Recorded.generations (Loop.root chosen))
                    pure $ do
                        (transcribed, observed) <- first (Recovery . ("A transcript or a generation cannot be read: " ++) . show) found
                        (replayed, commands) <- first Recovery (Replay.resume replaying later transcribed observed)
                        pure (Resumed started selected (Replay.plan replaying) observed (origin chosen replayed commands))
    _ -> pure (Left (Recovery "The journal does not start with a run declaration"))

interpret :: FilePath -> (Value -> Either String (Loop.Config, Natural)) -> Object -> IO (Either Error Run)
interpret target interpreter declared = case parseEither (\fields -> (,) <$> fields .: "arguments" <*> fields .: "workload") declared of
    Left problem -> pure (Left (Declaration problem))
    Right (arguments, workload) -> case (,) <$> interpreter arguments <*> Workload.decode (Lazy.toStrict (encode (workload :: Value))) of
        Left problem -> pure (Left (Declaration problem))
        Right ((chosen, lag), document) -> do
            same <- (== target) <$> canonicalizePath (Loop.root chosen)
            pure (if same then Right (Run chosen lag document) else Left (Declaration "The declared output directory is not the run's directory"))

origin :: Loop.Config -> Replay.Replayed -> [Command] -> Origin
origin chosen replayed commands = Origin (Replay.state replayed) commands versions (Replay.epoch floors) (Replay.identity floors) (Replay.numbered floors)
  where
    floors = Replay.floors replayed
    versions = Map.mapWithKey (\version (policy, learner, described) -> Published (Loop.Checkpoint (located version) policy learner) described) (Replay.versions replayed)
    located version = if version == 0 then Loop.checkpoint chosen else Loop.root chosen </> ("generation" ++ show version)

proceed :: Resumed -> Journal.Journal -> IO (Either Error ())
proceed (Resumed started selected planned observed restarted@(Origin state _ _ _ _ _)) recorded =
    withCurrentDirectory started $ do
        Journal.append recorded (Entry.encode (Entry.Restarted observed))
        announce (object ["phase" .= ("resumed" :: String), "committed" .= [update | Update update <- Core.committed state]])
        if Core.committed state == Plan.updates planned then pure (Right ()) else begin selected planned restarted recorded

declaration :: Run -> IO (Either Error Replay.Declaration)
declaration selected@(Run chosen lag _) = fmap (\(planned, Published _ description) -> Replay.Declaration chosen lag planned (instantiated selected) (scanl (+) 0 (counts selected)) description) <$> prepare selected

instantiated :: Run -> [String -> Either String Loop.Cycle]
instantiated (Run chosen _ document) = [\policy -> Loop.instantiate (policy, L.tokenizer settings, L.behaviorBase settings, L.behaviorAssembly settings) declared | declared <- Workload.cycles document]
  where
    settings = Loop.settings chosen

counts :: Run -> [Natural]
counts (Run _ _ document) = [genericLength (Workload.tasks declared) | declared <- Workload.cycles document]

prepare :: Run -> IO (Either Error (Plan.Plan, Published))
prepare selected@(Run chosen lag _)
    | Loop.inferenceMode (Loop.backend chosen) == R.Shared || Loop.learningMode (Loop.backend chosen) == W.Shared = pure (Left (Declaration "Concurrent rollout and learning need separate inference and learning processes"))
    | otherwise = do
        description <- Policy.readDescription (Loop.checkpoint chosen </> "policy.json")
        let settings = Loop.settings chosen
            initial = Loop.Checkpoint (Loop.checkpoint chosen) (L.policy settings) (L.learner settings)
            sizes = counts selected
            members = [[Request (offset + index) | index <- genericTake size [0 ..]] | (offset, size) <- zip (scanl (+) 0 sizes) sizes]
            expected = (L.policy settings, L.tokenizer settings, L.behaviorBase settings, L.behaviorAssembly settings)
        pure $ do
            unless (Policy.bindings description == expected) (Left (Declaration "Initial policy description differs from the declared inference materialization"))
            mapM_ (first Declaration . ($ L.policy settings)) (instantiated selected)
            planned <- first (Declaration . show) (Plan.prepare lag [Declared requests (chunks (L.steps settings) requests) | requests <- members])
            pure (planned, Published initial description)

begin :: Run -> Plan.Plan -> Origin -> Journal.Journal -> IO (Either Error ())
begin selected@(Run chosen lag _) planned (Origin state commands versions next identities numbers) recorded = do
    let settings = Loop.settings chosen
        engine = Loop.backend chosen
        initial = Loop.Checkpoint (Loop.checkpoint chosen) (L.policy settings) (L.learner settings)
    counted <- newMVar numbers
    linked <- newMVar Map.empty
    prepared <- newMVar Nothing
    owner <- newMVar Nothing
    let recording = Recorder recorded (Loop.root chosen </> "transcripts") counted next linked prepared owner
    outcome <- R.withRecordedDriver (Loop.inferenceMode engine) (Loop.inferenceWorker chosen initial, Loop.sessions engine) (inferenceTranscript recording) $ \collector -> do
        void (Internal.reserve collector identities)
        owned <- Owner.withRecordedRunner (Loop.learningMode engine) (Loop.updateWorker chosen (initial, 0)) (learnerTranscript recording (Loop.learningMode engine)) $ \updater -> do
            shared <-
                Shared chosen lag planned (instantiated selected) (scanl (+) 0 (counts selected)) recording
                    <$> newMVar state
                    <*> newMVar versions
                    <*> newMVar Map.empty
                    <*> newMVar Map.empty
                    <*> newChan
                    <*> newChan
                    <*> newEmptyMVar
                    <*> newMVar Nothing
                    <*> pure collector
                    <*> pure updater
            execute shared commands
        pure (either (Left . Learning) id owned)
    pure (either (Left . Rollout) id outcome)

reserve :: Recorder -> Entry.Role -> Natural -> IO Natural
reserve recording role slot = do
    number <- modifyMVar (numbers recording) (\next -> pure (next + 1, next))
    Journal.append (journal recording) (Entry.encode (Entry.Reserved number role slot (epoch recording)))
    pure number

transcribe :: Recorder -> Natural -> IO Transcript.Transcript
transcribe recording number = Journal.transcript (journal recording) (transcripts recording </> (show number ++ ".jsonl")) (Entry.encode . Entry.Finished number)

inferenceTranscript :: Recorder -> Natural -> IO Transcript.Transcript
inferenceTranscript recording slot = do
    number <- reserve recording Entry.Inference slot
    modifyMVar_ (processOf recording) (pure . Map.insert slot number)
    transcribe recording number

learnerTranscript :: Recorder -> W.Mode -> IO Transcript.Transcript
learnerTranscript recording mode = case mode of
    W.Resident -> do
        number <- reserve recording Entry.Learner 0
        modifyMVar_ (owning recording) (const (pure (Just number)))
        transcribe recording number
    _ -> modifyMVar (launching recording) (maybe (ioError (userError "A learner process was launched without its transcript")) (\opened -> pure (Nothing, opened)))

learnerProcess :: Recorder -> W.Mode -> IO Natural
learnerProcess recording mode = case mode of
    W.Resident -> readMVar (owning recording) >>= maybe (ioError (userError "The resident learner owner was never reserved")) pure
    _ -> do
        number <- reserve recording Entry.Learner 0
        opened <- transcribe recording number
        modifyMVar_ (launching recording) (const (pure (Just opened)))
        pure number

execute :: Shared scope -> [Command] -> IO (Either Error ())
execute shared initial =
    bracket (worker (forever (readChan (cohorts shared) >>= guarded shared . collect shared))) stop $ \_ ->
        bracket (worker (forever (readChan (updates shared) >>= guarded shared . learn shared))) stop $ \_ -> do
            guarded shared $ do
                forM_ (genericTake sessions [0 ..]) $ \slot -> do
                    Journal.append (journal (recorder shared)) (Entry.encode (Entry.Opened slot (epoch (recorder shared))))
                    void (transition shared (Connected (Worker slot) (Epoch (epoch (recorder shared)))))
                route shared initial
                finish shared
            readMVar (finished shared)
  where
    sessions = fromIntegral (length (Loop.sessions (Loop.backend (config shared)))) :: Natural
    worker action = do
        exited <- newEmptyMVar
        thread <- forkFinally action (\outcome -> either (fail' shared . Crashed . show) pure outcome `finally` putMVar exited ())
        pure (thread, exited)
    stop (thread, exited) = killThread thread >> readMVar exited

guarded :: Shared scope -> IO () -> IO ()
guarded shared action = do
    outcome <- try action
    case outcome of
        Right () -> pure ()
        Left problem
            | Just cancelled <- fromException problem -> throwIO (cancelled :: SomeAsyncException)
            | otherwise -> fail' shared (Crashed (show problem))

finish :: Shared scope -> IO ()
finish shared = do
    state <- readMVar (core shared)
    when (Core.committed state == Plan.updates (plan shared)) (void (tryPutMVar (finished shared) (Right ())))

-- The core reports an attempt in flight, but only the runtime knows whether its worker may have run, which is what leaves the attempt's effects unknown; the index names that update, which at a positive staleness may not be the update the failure came from.
fail' :: Shared scope -> Error -> IO ()
fail' shared problem = do
    running <- readMVar (invoked shared)
    void (tryPutMVar (finished shared) (Left (maybe problem (`Unresolved` problem) running)))

transition :: Shared scope -> Event -> IO [Command]
transition shared event = do
    returned <- modifyMVar (core shared) $ \state -> case Core.step state event of
        Left problem -> pure (state, Left problem)
        Right (next, commands) -> do
            Journal.append (journal (recorder shared)) (Entry.encode (Entry.Happened (Entry.claim event)))
            pure (next, Right commands)
    case returned of
        Left problem -> ioError (userError ("Event core refused " ++ show event ++ ": " ++ show problem))
        Right commands -> do
            route shared commands
            pure commands

route :: Shared scope -> [Command] -> IO ()
route shared = mapM_ (routed shared)

routed :: Shared scope -> Command -> IO ()
routed shared issued = case issued of
    Dispatch request (Version version) ->
        case Plan.owner (plan shared) request of
            Nothing -> ioError (userError "Dispatched request has no owning update")
            Just (Update update) -> do
                let expected = maybe [] Plan.members (Plan.declared (plan shared) (Update update))
                complete <- modifyMVar (pending shared) $ \waiting -> do
                    let collected = Map.findWithDefault [] update waiting ++ [request]
                    pure (if length collected == length expected then (Map.delete update waiting, True) else (Map.insert update collected waiting, False))
                when complete (writeChan (cohorts shared) (update, version))
    Send (Update update) attempt -> writeChan (updates shared) (update, attempt)
    _ -> pure ()

collect :: Shared scope -> (Natural, Natural) -> IO ()
collect shared (update, version) = do
    Published checkpoint description <- (Map.! version) <$> readMVar (published shared)
    started <- getMonotonicTime
    slots <- newIORef Map.empty
    cycle' <- either (ioError . userError) pure ((cycles shared !! fromIntegral update) (Loop.policy checkpoint))
    tasks <- either (ioError . userError . show) pure (Loop.bindTasks description (Loop.tasks cycle'))
    let chosen = config shared
        offset = offsets shared !! fromIntegral update
        options = R.Options {R.worker = Loop.inferenceWorker chosen checkpoint, R.mode = Loop.inferenceMode (Loop.backend chosen), R.sessions = Loop.sessions (Loop.backend chosen), R.definition = C.Definition (Loop.policy checkpoint) tasks, R.order = Loop.order cycle', R.delivery = Loop.delivery cycle', R.reference = Loop.scoring chosen checkpoint}
        observer = R.Observer (dispatched slots offset) (checked slots offset)
    generated <- R.runObserved (driver shared) observer options
    case generated of
        Left problem -> fail' shared (Rollout problem)
        Right batch -> do
            interval shared ("rollout", update) started
            box <- modifyMVar (batches shared) $ \held -> case Map.lookup update held of
                Just existing -> pure (held, existing)
                Nothing -> newEmptyMVar >>= \fresh -> pure (Map.insert update fresh held, fresh)
            putMVar box batch
  where
    dispatched slots offset slot requests = do
        number <- maybe (ioError (userError "Requests were dispatched to a slot without a process")) pure . Map.lookup slot =<< readMVar (processOf (recorder shared))
        forM_ requests $ \(index, binding) -> do
            atomicModifyIORef' slots (\held -> (Map.insert index slot held, ()))
            Journal.append (journal (recorder shared)) (Entry.encode (Entry.Dispatched (offset + index) slot (epoch (recorder shared)) binding number))
        forM_ requests $ \(index, _) -> void (transition shared (Started (Worker slot) (Epoch (epoch (recorder shared))) (Request (offset + index))))
    checked slots offset index admitted = do
        slot <- maybe (ioError (userError "Checked result was never dispatched")) pure . Map.lookup index =<< readIORef slots
        let request = offset + index
            digest = Trajectory.digest admitted
        Journal.append (journal (recorder shared)) (Entry.encode (Entry.Stored request (Trajectory.binding admitted) digest))
        void (transition shared (Completed (Worker slot) (Epoch (epoch (recorder shared))) (Request request) digest))

learn :: Shared scope -> (Natural, Attempt) -> IO ()
learn shared (update, attempt@(Attempt counter)) = do
    box <- modifyMVar (batches shared) $ \held -> case Map.lookup update held of
        Just existing -> pure (held, existing)
        Nothing -> newEmptyMVar >>= \fresh -> pure (Map.insert update fresh held, fresh)
    batch <- readMVar box
    versions <- readMVar (published shared)
    let version = if update > staleness shared then update - staleness shared else 0
        Published current description = versions Map.! update
        Published behavior _ = versions Map.! version
        chosen = config shared
        settings = (Loop.settings chosen) {L.policy = Loop.policy current, L.learner = Loop.learner current, L.schedule = L.Schedule update (staleness shared) version (Loop.policy behavior)}
    started <- getMonotonicTime
    case L.prepare settings batch of
        Left problem -> fail' shared (Admission problem)
        Right planned -> do
            ordinal <- Internal.reserve (driver shared) 1
            let binding = V.ordinal ordinal
                staging = Loop.stagedName ordinal
            number <- learnerProcess (recorder shared) (Loop.learningMode (Loop.backend chosen))
            Journal.append (journal (recorder shared)) (Entry.encode (Entry.Attempted update counter binding number))
            case W.prepare binding planned of
                Left problem -> fail' shared (Learning problem)
                Right call -> do
                    forwarded <- newIORef 0
                    opened <- newIORef Nothing
                    let hooks = W.Hooks (ready attempt) (replying attempt forwarded opened)
                        paths = Resident.Paths (Loop.directory current) (Loop.root chosen </> staging)
                    modifyMVar_ (invoked shared) (const (pure (Just update)))
                    executed <- Owner.run (runner shared) paths (W.hooked hooks call)
                    case executed of
                        Left problem -> fail' shared (Learning problem)
                        Right observed -> do
                            interval shared ("learner", update) started
                            let produced = Observation.report observed
                            close attempt forwarded (S.completions (P.stream produced))
                            Journal.append (journal (recorder shared)) (Entry.encode (Entry.Verified update counter (P.learner produced)))
                            commands <- transition shared (Staged (Update update) attempt (P.adapter produced))
                            publish shared (update, attempt, ordinal) (description, produced) commands
  where
    ready current stream = do
        void (transition shared (Ready (Update update) current (S.exchange stream) (S.identity stream) (S.opening stream)))
        pure (Right ())
    replying current forwarded opened stream reply = do
        close current forwarded (S.completions stream)
        let index = S.replyStep reply
        previous <- readIORef opened
        when (previous /= Just index) $ do
            commands <- transition shared (Current (Update update) current index (S.replyState reply))
            unless (Open (Update update) current index (S.replyState reply) `elem` commands) (ioError (userError "The event core did not open the reported step"))
            writeIORef opened (Just index)
        state <- readMVar (core shared)
        pure (either (Left . show) Right (Core.authorize state (Update update) current index))
    close current forwarded closed = do
        count <- readIORef forwarded
        forM_ (drop count closed) $ \done -> void (transition shared (Applied (Update update) current done))
        writeIORef forwarded (length closed)

publish :: Shared scope -> (Natural, Attempt, Natural) -> (Policy.Description, P.Result) -> [Command] -> IO ()
publish shared (update, attempt, ordinal) (description, produced) commands = do
    unless (any recording commands) (ioError (userError "The event core did not ask to record the staged update"))
    successor <- either (ioError . userError) pure (Policy.successor (P.adapter produced) description)
    let chosen = config shared
        staging = Loop.stagedName ordinal
        name = "generation" ++ show (update + 1)
    Policy.stageDescription (Loop.root chosen </> staging </> "policy.json") successor
    recorded <- transition shared (Recorded (Update update) attempt)
    unless (Commit (Update update) attempt `elem` recorded) (ioError (userError "The event core did not commit the recorded update"))
    receipt <- Store.publishCheckpoint (Loop.publication chosen) (Store.Location (Loop.root chosen) (Char.pack staging) (Char.pack name))
    let checkpoint = Loop.Checkpoint (Loop.root chosen </> name) (P.adapter produced) (P.learner produced)
    modifyMVar_ (published shared) (pure . Map.insert (update + 1) (Published checkpoint successor))
    announce (object ["phase" .= ("published" :: String), "update" .= update, "version" .= (update + 1), "checkpoint" .= Loop.directory checkpoint, "policy" .= Loop.policy checkpoint, "learner" .= Loop.learner checkpoint, "publication" .= Store.methodName (Store.method receipt)])
    void (transition shared (Committed (Update update) attempt))
    modifyMVar_ (invoked shared) (const (pure Nothing))
    modifyMVar_ (batches shared) (pure . Map.delete update)
    finish shared
  where
    recording (Record (Update target) current _) = target == update && current == attempt
    recording _ = False

interval :: Shared scope -> (Text, Natural) -> Double -> IO ()
interval shared (role, update) started = do
    ended <- getMonotonicTime
    Journal.append (journal (recorder shared)) (Entry.encode (Entry.Elapsed role update started ended))

announce :: Value -> IO ()
announce value = Transcript.live (Lazy.toStrict (encode value))

chunks :: Natural -> [value] -> [[value]]
chunks count = go (fromIntegral count)
  where
    go :: Int -> [value] -> [[value]]
    go 0 _ = []
    go remaining rest = let size = (length rest + remaining - 1) `div` remaining in take size rest : go (remaining - 1) (drop size rest)
