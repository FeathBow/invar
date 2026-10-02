{-# LANGUAGE OverloadedStrings #-}

module Invar.Runtime (Error (..), Run (..), run, resume) where

import Control.Concurrent (forkFinally, killThread)
import Control.Concurrent.Chan (Chan, newChan, readChan, writeChan)
import Control.Concurrent.MVar (MVar, modifyMVar, modifyMVar_, newEmptyMVar, newMVar, putMVar, readMVar, tryPutMVar)
import Control.Exception (SomeAsyncException, bracket, finally, fromException, throwIO, try)
import Control.Monad (foldM, forM_, forever, unless, void, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Object, Value, encode, object, withObject, (.:), (.=))
import Data.Aeson.Types (Pair, Parser, parseEither)
import Data.Bifunctor (first)
import Data.ByteString.Char8 qualified as Char
import Data.ByteString.Lazy qualified as Lazy
import Data.Char (isDigit)
import Data.IORef (atomicModifyIORef', newIORef, readIORef, writeIORef)
import Data.List (genericTake, sort, stripPrefix)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set (Set)
import Data.Set qualified as Set
import GHC.Clock (getMonotonicTime)
import Invar.Artifact qualified as Artifact
import Invar.Async.Completion qualified as Completion
import Invar.Async.Core (Attempt (..), Command (..), Epoch (..), Event (..), Worker (..))
import Invar.Async.Core qualified as Core
import Invar.Async.Plan (Declared (..), Request (..), Update (..), Version (..))
import Invar.Async.Plan qualified as Plan
import Invar.Cohort qualified as C
import Invar.Infer.Result qualified as Result
import Invar.Infer.Wire qualified as Wire
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
import Invar.Process qualified as Process
import Invar.Rollout qualified as R
import Invar.Rollout.Internal qualified as Internal
import Invar.Spec.Invocation qualified as V
import Invar.Store qualified as Store
import Numeric.Natural (Natural)
import System.Directory (canonicalizePath, createDirectory, getCurrentDirectory, listDirectory, withCurrentDirectory)
import System.FilePath ((</>))

data Error = Declaration String | Recovery String | Rollout R.Error | Admission L.Error | Learning W.Failure | Crashed String
    deriving (Show)

data Run = Run Loop.Config Natural [String -> Either String Loop.Cycle] [Natural]

data Published = Published Loop.Checkpoint Policy.Description

data Origin = Origin Core.State [Command] (Map Natural Published) Natural Natural

data Ledger = Ledger
    { acknowledged :: Set Natural
    , commits :: Map Natural Natural
    , staged :: Map (Natural, Natural) String
    , learners :: Map (Natural, Natural) String
    , epochs :: Map Natural Natural
    , bindings :: Set V.Binding
    , attempts :: Set Natural
    }

data Shared scope = Shared
    { config :: Loop.Config
    , staleness :: Natural
    , plan :: Plan.Plan
    , cycles :: [String -> Either String Loop.Cycle]
    , offsets :: [Natural]
    , journal :: Journal.Journal
    , epoch :: Natural
    , core :: MVar Core.State
    , published :: MVar (Map Natural Published)
    , batches :: MVar (Map Natural (MVar (R.Batch scope)))
    , pending :: MVar (Map Natural [Request])
    , cohorts :: Chan (Natural, Natural)
    , updates :: Chan (Natural, Attempt)
    , finished :: MVar (Either Error ())
    , driver :: R.Driver scope
    , runner :: Owner.Runner
    }

run :: Run -> [Pair] -> IO (Either Error ())
run selected@(Run chosen _ _ _) declared = do
    prepared <- prepare selected
    case prepared of
        Left problem -> pure (Left problem)
        Right (planned, initial) -> do
            started <- getCurrentDirectory
            let declaration = object (["entry" .= ("declaration" :: String), "directory" .= started] ++ declared)
            createDirectory (Loop.root chosen)
            createDirectory (Loop.root chosen </> "results")
            Journal.with (Loop.root chosen </> "journal.jsonl") declaration $ \recorded -> do
                let (state, commands) = Core.start planned
                begin selected planned (Origin state commands (Map.singleton 0 initial) 0 0) recorded

resume :: FilePath -> (Object -> Either String Run) -> IO (Either Error ())
resume directory interpret = do
    target <- canonicalizePath directory
    Journal.resume (target </> "journal.jsonl") (continue target interpret)

continue :: FilePath -> (Object -> Either String Run) -> [Object] -> Journal.Journal -> IO (Either Error ())
continue target interpret recorded resumed = case recorded of
    declaration : later | parseEither (.: "entry") declaration == Right ("declaration" :: String) -> case parseEither (.: "directory") declaration of
        Left problem -> pure (Left (Recovery problem))
        Right started ->
            withCurrentDirectory started $ case interpret declaration of
                Left problem -> pure (Left (Declaration problem))
                Right selected@(Run chosen _ _ _) -> do
                    same <- (== target) <$> canonicalizePath (Loop.root chosen)
                    prepared <- prepare selected
                    case prepared of
                        _ | not same -> pure (Left (Declaration "The declared output directory is not the resumed directory"))
                        Left problem -> pure (Left problem)
                        Right (planned, initial) -> do
                            recovered <- recover chosen planned initial later
                            case recovered of
                                Left problem -> pure (Left problem)
                                Right origin@(Origin state _ _ next identities) -> do
                                    let done = [update | Update update <- Core.committed state]
                                    Journal.append resumed (object ["entry" .= ("resume" :: String), "committed" .= done, "epoch" .= next, "identities" .= identities])
                                    announce (object ["phase" .= ("resumed" :: String), "committed" .= done])
                                    if Core.committed state == Plan.updates planned then pure (Right ()) else begin selected planned origin resumed
    _ -> pure (Left (Recovery "The journal does not start with a run declaration"))

prepare :: Run -> IO (Either Error (Plan.Plan, Published))
prepare (Run chosen lag _ sizes)
    | Loop.inferenceMode (Loop.backend chosen) == R.Shared || Loop.learningMode (Loop.backend chosen) == W.Shared = pure (Left (Declaration "Concurrent rollout and learning need separate inference and learning processes"))
    | otherwise = do
        description <- Policy.readDescription (Loop.checkpoint chosen </> "policy.json")
        let settings = Loop.settings chosen
            initial = Loop.Checkpoint (Loop.checkpoint chosen) (L.policy settings) (L.learner settings)
            members = [[Request (offset + index) | index <- genericTake size [0 ..]] | (offset, size) <- zip (scanl (+) 0 sizes) sizes]
            expected = (L.policy settings, L.tokenizer settings, L.behaviorBase settings, L.behaviorAssembly settings)
        pure $ do
            unless (Policy.bindings description == expected) (Left (Declaration "Initial policy description differs from the declared inference materialization"))
            planned <- first (Declaration . show) (Plan.prepare lag [Declared requests (chunks (L.steps settings) requests) | requests <- members])
            pure (planned, Published initial description)

recover :: Loop.Config -> Plan.Plan -> Published -> [Object] -> IO (Either Error Origin)
recover chosen planned initial later = case parseEither (foldM ledger (Ledger Set.empty Map.empty Map.empty Map.empty Map.empty Set.empty Set.empty)) later of
    Left problem -> pure (Left (Recovery problem))
    Right found -> do
        names <- listDirectory (Loop.root chosen)
        let generations = sort [read digits | name <- names, Just digits <- [stripPrefix "generation" name], not (null digits), all isDigit digits] :: [Natural]
            committed = fromIntegral (Set.size (acknowledged found))
            count = fromIntegral (length generations)
        case () of
            _
                | acknowledged found /= Set.fromList (genericTake committed [0 ..]) -> pure (Left (Recovery "Journaled commits are not the first updates in order"))
                | generations /= [1 .. count] -> pure (Left (Recovery "Published generations are not consecutive from the first"))
                | count < committed -> pure (Left (Recovery ("Update " ++ show count ++ " is committed in the journal but its generation is missing")))
                | count > committed + 1 || count > fromIntegral (length (Plan.updates planned)) -> pure (Left (Recovery "More generations are published than the journal committed"))
                | otherwise -> do
                    versions <- foldM (\held version -> either (pure . Left) (`generation` version) held) (Right (Map.singleton 0 initial)) [1 .. count]
                    pure $ do
                        held <- versions
                        let lowest = Map.fromList [(Worker worker, Epoch used) | (worker, used) <- Map.toList (epochs found)]
                            next = maybe 0 (+ 1) (Set.lookupMax (Set.fromList (Map.elems (epochs found))))
                            identities = maybe 0 (+ 1) (Set.lookupMax (Set.map largest (bindings found)))
                            fresh = maybe 0 (+ 1) (Set.lookupMax (attempts found))
                        (state, commands) <- first (Recovery . show) (Core.recover planned (map Update (genericTake count [0 ..])) Map.empty lowest (bindings found) fresh)
                        pure (Origin state commands held next identities)
      where
        generation held version = do
            let update = version - 1
                directory = Loop.root chosen </> ("generation" ++ show version)
                Published _ previous = held Map.! update
            description <- Policy.readDescription (directory </> "policy.json")
            adapter <- Policy.identity (directory </> "adapter.safetensors")
            learner <- Artifact.identity "Learner checkpoint" (directory </> "learner.pt")
            let journaled = Map.lookup update (commits found) >>= \attempt -> (,) <$> Map.lookup (update, attempt) (staged found) <*> Map.lookup (update, attempt) (learners found)
            pure $ case journaled of
                Nothing -> Left (Recovery ("Generation " ++ show version ++ " has no journaled commit request"))
                Just expected
                    | expected /= (adapter, learner) || Policy.successor adapter previous /= Right description -> Left (Recovery ("Generation " ++ show version ++ " differs from the update its attempt staged"))
                    | otherwise -> Right (Map.insert version (Published (Loop.Checkpoint directory adapter learner) description) held)

largest :: V.Binding -> Natural
largest (V.Binding (V.CallId call) (V.AttemptId attempt) (V.Instance instanceId)) = maximum [call, attempt, instanceId]

ledger :: Ledger -> Object -> Parser Ledger
ledger found fields = do
    kind <- fields .: "entry"
    case kind :: String of
        "epoch" -> do
            worker <- fields .: "worker"
            used <- fields .: "epoch"
            pure found {epochs = Map.insertWith max worker used (epochs found)}
        "dispatched" -> do
            bound <- Wire.binding fields
            pure found {bindings = Set.insert bound (bindings found)}
        "attempt" -> do
            bound <- Wire.binding fields
            attempt <- fields .: "attempt"
            pure found {bindings = Set.insert bound (bindings found), attempts = Set.insert attempt (attempts found)}
        "event" -> do
            recorded <- fields .: "event" >>= withObject "journaled event" (journaled found)
            issued <- fields .: "commands" :: Parser [Value]
            foldM (withObject "journaled command" . commanded) recorded issued
        "checkpoint" -> do
            key <- (,) <$> fields .: "update" <*> fields .: "attempt"
            digest <- fields .: "learner"
            pure found {learners = Map.insert key digest (learners found)}
        "resume" -> do
            done <- fields .: "committed"
            pure found {acknowledged = Set.union (acknowledged found) (Set.fromList done)}
        _ | kind `elem` ["result", "interval"] -> pure found
        _ -> fail ("Unknown journal entry: " ++ kind)
  where
    journaled held event = do
        kind <- event .: "kind"
        case kind :: String of
            "committed" -> do
                update <- event .: "update"
                pure held {acknowledged = Set.insert update (acknowledged held)}
            "staged" -> do
                key <- (,) <$> event .: "update" <*> event .: "attempt"
                digest <- event .: "digest"
                pure held {staged = Map.insert key digest (staged held)}
            _ -> pure held
    commanded held command' = do
        kind <- command' .: "kind"
        case kind :: String of
            "send" -> do
                attempt <- command' .: "attempt"
                pure held {attempts = Set.insert attempt (attempts held)}
            "commit" -> do
                update <- command' .: "update"
                attempt <- command' .: "attempt"
                pure held {commits = Map.insert update attempt (commits held)}
            _ -> pure held

begin :: Run -> Plan.Plan -> Origin -> Journal.Journal -> IO (Either Error ())
begin (Run chosen lag workload sizes) planned (Origin state commands versions next identities) recorded = do
    let settings = Loop.settings chosen
        engine = Loop.backend chosen
        initial = Loop.Checkpoint (Loop.checkpoint chosen) (L.policy settings) (L.learner settings)
    outcome <- R.withConfiguredDriver (Loop.inferenceMode engine) (Loop.inferenceWorker chosen initial, Loop.sessions engine) $ \collector -> do
        void (Internal.reserve collector identities)
        owned <- Owner.withRunner (Loop.learningMode engine) (Loop.updateWorker chosen (initial, 0)) $ \updater -> do
            shared <-
                Shared chosen lag planned workload (scanl (+) 0 sizes) recorded next
                    <$> newMVar state
                    <*> newMVar versions
                    <*> newMVar Map.empty
                    <*> newMVar Map.empty
                    <*> newChan
                    <*> newChan
                    <*> newEmptyMVar
                    <*> pure collector
                    <*> pure updater
            execute shared commands
        pure (either (Left . Learning) id owned)
    pure (either (Left . Rollout) id outcome)

execute :: Shared scope -> [Command] -> IO (Either Error ())
execute shared initial =
    bracket (worker (forever (readChan (cohorts shared) >>= guarded shared . collect shared))) stop $ \_ ->
        bracket (worker (forever (readChan (updates shared) >>= guarded shared . learn shared))) stop $ \_ -> do
            guarded shared $ do
                forM_ (genericTake sessions [0 ..]) $ \slot -> do
                    Journal.append (journal shared) (object ["entry" .= ("epoch" :: String), "worker" .= slot, "epoch" .= epoch shared])
                    void (transition shared (Connected (Worker slot) (Epoch (epoch shared))))
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

fail' :: Shared scope -> Error -> IO ()
fail' shared problem = void (tryPutMVar (finished shared) (Left problem))

transition :: Shared scope -> Event -> IO [Command]
transition shared event = do
    returned <- modifyMVar (core shared) $ \state -> case Core.step state event of
        Left problem -> pure (state, Left problem)
        Right (next, commands) -> do
            Journal.append (journal shared) (object ["entry" .= ("event" :: String), "event" .= eventValue event, "commands" .= map commandValue commands])
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
        forM_ requests $ \(index, binding) -> do
            atomicModifyIORef' slots (\held -> (Map.insert index slot held, ()))
            Journal.append (journal shared) (object ["entry" .= ("dispatched" :: String), "request" .= (offset + index), "worker" .= slot, "epoch" .= epoch shared, "binding" .= Wire.bindingValue binding])
        forM_ requests $ \(index, _) -> void (transition shared (Started (Worker slot) (Epoch (epoch shared)) (Request (offset + index))))
    checked slots offset index binding result = do
        slot <- maybe (ioError (userError "Checked result was never dispatched")) pure . Map.lookup index =<< readIORef slots
        let request = offset + index
            V.Binding (V.CallId call) _ _ = binding
            encoded = Lazy.toStrict (encode (object ["binding" .= Wire.bindingValue binding, "result" .= Result.record result]))
            path = Loop.root (config shared) </> "results" </> ("call" ++ show call ++ ".json")
        Journal.store path encoded
        let digest = Artifact.hex (SHA256.hash encoded)
        Journal.append (journal shared) (object ["entry" .= ("result" :: String), "request" .= request, "binding" .= Wire.bindingValue binding, "digest" .= digest])
        void (transition shared (Completed (Worker slot) (Epoch (epoch shared)) (Request request) digest))

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
            Journal.append (journal shared) (object ["entry" .= ("attempt" :: String), "update" .= update, "attempt" .= counter, "binding" .= Wire.bindingValue binding])
            case W.prepare binding planned of
                Left problem -> fail' shared (Learning problem)
                Right call -> do
                    forwarded <- newIORef 0
                    opened <- newIORef Nothing
                    let hooks = W.Hooks (ready attempt) (replying attempt forwarded opened)
                        paths = Resident.Paths (Loop.directory current) (Loop.root chosen </> staging)
                    executed <- Owner.run (runner shared) paths (W.hooked hooks call)
                    case executed of
                        Left problem -> fail' shared (Learning problem)
                        Right observed -> do
                            interval shared ("learner", update) started
                            let produced = Observation.report observed
                            close attempt forwarded (S.completions (P.stream produced))
                            Journal.append (journal shared) (object ["entry" .= ("checkpoint" :: String), "update" .= update, "attempt" .= counter, "learner" .= P.learner produced])
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
    modifyMVar_ (batches shared) (pure . Map.delete update)
    finish shared
  where
    recording (Record (Update target) current _) = target == update && current == attempt
    recording _ = False

interval :: Shared scope -> (String, Natural) -> Double -> IO ()
interval shared (role, update) started = do
    ended <- getMonotonicTime
    Journal.append (journal shared) (object ["entry" .= ("interval" :: String), "role" .= role, "update" .= update, "start" .= started, "end" .= ended])

announce :: Value -> IO ()
announce value = Process.live (Lazy.toStrict (encode value))

eventValue :: Event -> Value
eventValue event = case event of
    Connected (Worker worker) (Epoch used) -> object ["kind" .= ("connected" :: String), "worker" .= worker, "epoch" .= used]
    Lost (Worker worker) (Epoch used) -> object ["kind" .= ("lost" :: String), "worker" .= worker, "epoch" .= used]
    Started (Worker worker) (Epoch used) (Request request) -> object ["kind" .= ("started" :: String), "worker" .= worker, "epoch" .= used, "request" .= request]
    Completed (Worker worker) (Epoch used) (Request request) digest -> object ["kind" .= ("completed" :: String), "worker" .= worker, "epoch" .= used, "request" .= request, "digest" .= digest]
    Ready update attempt bound identity before -> addressed "ready" update attempt ["binding" .= Wire.bindingValue bound, "plan" .= identity, "before" .= before]
    Current update attempt index before -> addressed "current" update attempt ["step" .= index, "before" .= before]
    Applied update attempt done -> addressed "applied" update attempt ["binding" .= Wire.bindingValue (Completion.binding done), "plan" .= Completion.plan done, "step" .= Completion.step done, "consumed" .= Completion.consumed done, "before" .= Completion.before done, "after" .= Completion.after done]
    Staged update attempt digest -> addressed "staged" update attempt ["digest" .= digest]
    Recorded update attempt -> addressed "recorded" update attempt []
    Committed update attempt -> addressed "committed" update attempt []
    Abandoned update attempt -> addressed "abandoned" update attempt []

commandValue :: Command -> Value
commandValue issued = case issued of
    Dispatch (Request request) (Version version) -> object ["kind" .= ("dispatch" :: String), "request" .= request, "version" .= version]
    Send update attempt -> addressed "send" update attempt []
    Open update attempt index before -> addressed "open" update attempt ["step" .= index, "before" .= before]
    Record update attempt digest -> addressed "record" update attempt ["digest" .= digest]
    Commit update attempt -> addressed "commit" update attempt []

addressed :: String -> Update -> Attempt -> [Pair] -> Value
addressed kind (Update update) (Attempt attempt) rest = object (["kind" .= kind, "update" .= update, "attempt" .= attempt] ++ rest)

chunks :: Natural -> [value] -> [[value]]
chunks count = go (fromIntegral count)
  where
    go :: Int -> [value] -> [[value]]
    go 0 _ = []
    go remaining rest = let size = (length rest + remaining - 1) `div` remaining in take size rest : go (remaining - 1) (drop size rest)
