{-# LANGUAGE OverloadedStrings #-}

module Invar.Runtime (Error (..), run) where

import Control.Concurrent (forkFinally, killThread)
import Control.Concurrent.Chan (Chan, newChan, readChan, writeChan)
import Control.Concurrent.MVar (MVar, modifyMVar, modifyMVar_, newEmptyMVar, newMVar, putMVar, readMVar, tryPutMVar)
import Control.Exception (SomeAsyncException, bracket, finally, fromException, throwIO, try)
import Control.Monad (forM_, forever, unless, void, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value, encode, object, toJSON, (.=))
import Data.ByteString.Char8 qualified as Char
import Data.ByteString.Lazy qualified as Lazy
import Data.IORef (atomicModifyIORef', newIORef, readIORef, writeIORef)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import GHC.Clock (getMonotonicTime)
import Invar.Artifact qualified as Artifact
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
import System.Directory (createDirectory)
import System.FilePath ((</>))

data Error = Declaration String | Rollout R.Error | Admission L.Error | Learning W.Failure | Crashed String
    deriving (Show)

data Published = Published Loop.Checkpoint Policy.Description

data Shared scope = Shared
    { config :: Loop.Config
    , staleness :: Natural
    , plan :: Plan.Plan
    , cycles :: [String -> Either String Loop.Cycle]
    , offsets :: [Natural]
    , journal :: Journal.Journal
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

run :: Loop.Config -> Natural -> Value -> [String -> Either String Loop.Cycle] -> [Natural] -> IO (Either Error ())
run chosen lag declaration workload sizes
    | Loop.inferenceMode (Loop.backend chosen) == R.Shared || Loop.learningMode (Loop.backend chosen) == W.Shared = pure (Left (Declaration "Concurrent rollout and learning need separate inference and learning processes"))
    | otherwise = do
        description <- Policy.readDescription (Loop.checkpoint chosen </> "policy.json")
        let settings = Loop.settings chosen
            engine = Loop.backend chosen
            initial = Loop.Checkpoint (Loop.checkpoint chosen) (L.policy settings) (L.learner settings)
            members = [[Request (offset + index) | index <- [0 .. size - 1]] | (offset, size) <- zip (scanl (+) 0 sizes) sizes]
        case Plan.prepare lag [Declared requests (chunks (L.steps settings) requests) | requests <- members] of
            Left problem -> pure (Left (Declaration (show problem)))
            Right plan -> do
                createDirectory (Loop.root chosen)
                createDirectory (Loop.root chosen </> "results")
                Journal.with (Loop.root chosen </> "journal.jsonl") declaration $ \recorded -> do
                    outcome <- R.withConfiguredDriver (Loop.inferenceMode engine) (Loop.inferenceWorker chosen initial, Loop.sessions engine) $ \collector -> do
                        owned <- Owner.withRunner (Loop.learningMode engine) (Loop.updateWorker chosen (initial, 0)) $ \updater -> do
                            shared <-
                                Shared chosen lag plan workload (scanl (+) 0 sizes) recorded
                                    <$> newMVar (fst (Core.start plan))
                                    <*> newMVar (Map.singleton 0 (Published initial description))
                                    <*> newMVar Map.empty
                                    <*> newMVar Map.empty
                                    <*> newChan
                                    <*> newChan
                                    <*> newEmptyMVar
                                    <*> pure collector
                                    <*> pure updater
                            execute shared
                        pure (either (Left . Learning) id owned)
                    pure (either (Left . Rollout) id outcome)

execute :: Shared scope -> IO (Either Error ())
execute shared =
    bracket (worker (forever (readChan (cohorts shared) >>= guarded shared . collect shared))) stop $ \_ ->
        bracket (worker (forever (readChan (updates shared) >>= guarded shared . learn shared))) stop $ \_ -> do
            guarded shared $ do
                forM_ [0 .. sessions - 1] $ \slot -> do
                    Journal.append (journal shared) (object ["entry" .= ("epoch" :: String), "worker" .= slot, "epoch" .= (0 :: Int)])
                    void (transition shared (Connected (Worker (fromIntegral slot)) (Epoch 0)))
                route shared (snd (Core.start (plan shared)))
                finish shared
            readMVar (finished shared)
  where
    sessions = length (Loop.sessions (Loop.backend (config shared)))
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
            Journal.append (journal shared) (object ["entry" .= ("event" :: String), "event" .= describe event, "commands" .= map command commands])
            pure (next, Right commands)
    case returned of
        Left problem -> ioError (userError ("Event core refused " ++ show (describe event) ++ ": " ++ show problem))
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
    dispatched slots offset slot requests = forM_ requests $ \(index, _) -> do
        atomicModifyIORef' slots (\held -> (Map.insert index slot held, ()))
        void (transition shared (Started (Worker slot) (Epoch 0) (Request (offset + index))))
    checked slots offset index binding result = do
        slot <- maybe (ioError (userError "Checked result was never dispatched")) pure . Map.lookup index =<< readIORef slots
        let request = offset + index
            encoded = Lazy.toStrict (encode (object ["binding" .= Wire.bindingValue binding, "result" .= Result.record result]))
            path = Loop.root (config shared) </> "results" </> ("request" ++ show request ++ ".json")
        Journal.store path encoded
        let digest = Artifact.hex (SHA256.hash encoded)
        Journal.append (journal shared) (object ["entry" .= ("result" :: String), "request" .= request, "digest" .= digest])
        void (transition shared (Completed (Worker slot) (Epoch 0) (Request request) digest))

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

describe :: Event -> Value
describe event = toJSON (show event)

command :: Command -> Value
command = toJSON . show

chunks :: Natural -> [value] -> [[value]]
chunks count = go (fromIntegral count)
  where
    go :: Int -> [value] -> [[value]]
    go 0 _ = []
    go remaining rest = let size = (length rest + remaining - 1) `div` remaining in take size rest : go (remaining - 1) (drop size rest)
