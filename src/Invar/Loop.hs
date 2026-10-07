{-# LANGUAGE RoleAnnotations #-}

module Invar.Loop (Backend (..), Config (..), Cycle (..), Checkpoint (..), Driver, Generation, Status (..), Error (..), withDriver, run, status, current, result, plan, receipt, inferenceWorker, updateWorker, scoring, stagedName, bindTasks, instantiate) where

import Control.Concurrent.MVar (MVar, newMVar, withMVar)
import Control.Exception (mask, onException)
import Control.Monad (join, unless)
import Data.Bifunctor (first)
import Data.ByteString.Char8 qualified as Bytes
import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import Invar.Cohort qualified as C
import Invar.Infer qualified as I
import Invar.Infer.Batch qualified as Batch
import Invar.Learn qualified as L
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Worker qualified as W
import Invar.Learn.Worker.Observation qualified as Observation
import Invar.Learn.Worker.Owner qualified as Owner
import Invar.Learn.Worker.Resident qualified as Resident
import Invar.Loop.Owner qualified as Shared
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as R
import Invar.Rollout.Internal qualified as Internal
import Invar.Spec.Invocation qualified as V
import Invar.Store qualified as S
import Invar.Worker qualified as Infer
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)
import System.Directory (createDirectory)
import System.FilePath ((</>))

data Backend = Backend
    { python :: FilePath
    , inferencePython :: FilePath
    , inference :: FilePath
    , inferenceConfiguration :: Maybe FilePath
    , inferenceMode :: R.Mode
    , learning :: FilePath
    , learningMode :: W.Mode
    , cache :: FilePath
    , sessions :: [[(String, String)]]
    }
data Config = Config {backend :: Backend, root :: FilePath, checkpoint :: FilePath, reference :: FilePath, settings :: L.Settings, publication :: S.Method}
data Cycle = Cycle {tasks :: [C.Task], order :: [Natural], delivery :: [Natural]}
data Checkpoint = Checkpoint {directory :: FilePath, policy :: String, learner :: String}
    deriving (Eq, Show)

type role Driver nominal
data Driver scope = Driver {configuration :: Config, rollout :: R.Driver scope, updater :: Owner.Runner, lock :: MVar (), state :: IORef (State scope)}

type role Generation nominal
data Generation scope = Generation Checkpoint (Observation.Observation scope) S.Receipt

data Cursor = Cursor Natural Checkpoint Policy.Description
data State scope = Idle Cursor | Collecting Cursor | Updating Cursor V.Binding | Publishing Cursor (Observation.Observation scope)

data Status = Ready Natural Checkpoint | Rolling Natural | Unresolved Natural V.Binding | Committing Natural V.Binding
    deriving (Eq, Show)

data Error = Settings L.Error | NotReady Status | Rollout R.Error | Plan L.Error | Update W.Failure | Policy String
    deriving (Eq, Show)

withDriver :: Config -> (forall scope. Driver scope -> IO value) -> IO (Either Error value)
withDriver config action = case L.validate (settings config) of
    Left problem -> pure (Left (Settings problem))
    Right () -> do
        selected <- Policy.readDescription (checkpoint config </> "policy.json")
        let chosen = settings config
            expected = (L.policy chosen, L.tokenizer chosen, L.behaviorBase chosen, L.behaviorAssembly chosen)
        if Policy.bindings selected /= expected
            then pure (Left (Policy "Initial policy description differs from the declared inference materialization"))
            else withPolicy config selected action

withPolicy :: Config -> Policy.Description -> (forall scope. Driver scope -> IO value) -> IO (Either Error value)
withPolicy config description action
    | inferenceMode (backend config) == R.Shared || learningMode (backend config) == W.Shared =
        if inferenceMode (backend config) /= R.Shared || learningMode (backend config) /= W.Shared
            then pure (Left (Update (W.ProtocolFailure "Both numerical roles must explicitly select shared execution")))
            else case Shared.validate shared of
                Left problem -> pure (Left (Update problem))
                Right () -> do
                    createDirectory (root config)
                    first Update <$> Shared.withShared shared (\pool runner -> R.withDriver (\(Internal.Driver gate counter _ recorded) -> drive (Internal.Driver gate counter (Just pool) recorded) runner))
    | otherwise = do
        createDirectory (root config)
        join . first Rollout
            <$> R.withConfiguredDriver
                (inferenceMode (backend config))
                (inferenceWorker config initial, sessions (backend config))
                ( \collector ->
                    first Update
                        <$> Owner.withRunner
                            (learningMode (backend config))
                            (updateWorker config (initial, 0))
                            (drive collector)
                )
  where
    initialGeneration = 0
    initial = Checkpoint (checkpoint config) (L.policy (settings config)) (L.learner (settings config))
    shared = (inferenceWorker config initial, updateWorker config (initial, 0), sessions (backend config))
    drive collector runner = do
        gate <- newMVar ()
        position <- newIORef (Idle (Cursor initialGeneration initial description))
        action (Driver config collector runner gate position)

run :: Driver scope -> Cycle -> IO (Either Error (Generation scope))
run driver workload = withMVar (lock driver) $ \() -> do
    previous <- readIORef (state driver)
    case previous of
        Idle cursor -> start driver (cursor, workload)
        _ -> pure (Left (NotReady (describe previous)))

start :: Driver scope -> (Cursor, Cycle) -> IO (Either Error (Generation scope))
start driver (cursor, workload) = mask $ \restore -> do
    writeIORef (state driver) (Collecting cursor)
    planned <- restore (collect driver (cursor, workload)) `onException` reset driver cursor
    case planned of
        Left problem -> reset driver cursor >> pure (Left problem)
        Right plannedUpdate -> execute driver (cursor, plannedUpdate) restore

collect :: Driver scope -> (Cursor, Cycle) -> IO (Either Error (L.Plan scope))
collect driver (cursor@(Cursor _ selected description), workload) = do
    actual <- Policy.readDescription (directory selected </> "policy.json")
    case prepare actual of
        Left problem -> pure (Left problem)
        Right boundTasks -> do
            let engine = backend config
                worker = inferenceWorker config selected
                options = R.Options {R.worker = worker, R.mode = inferenceMode engine, R.sessions = sessions engine, R.definition = C.Definition (policy selected) boundTasks, R.order = order workload, R.delivery = delivery workload, R.reference = scoring config selected}
            generated <- R.run (rollout driver) options
            pure $ do
                batch <- first Rollout generated
                first Plan (L.prepare (learnerSettings config cursor) batch)
  where
    config = configuration driver
    prepare actual = do
        first Plan (mapM_ (L.materialization (settings config) . I.requested . C.plan) (tasks workload))
        unless (actual == description) (Left (Policy "Selected policy description changed after publication or initial selection"))
        bindTasks description (tasks workload)

bindTasks :: Policy.Description -> [C.Task] -> Either Error [C.Task]
bindTasks description = traverse bind
  where
    bind task = do
        planned <- first (Policy . show) (I.bindPolicy description (C.plan task))
        pure task {C.plan = planned}

instantiate :: (String, String, String, String) -> Workload.Cycle -> Either String Cycle
instantiate (policy', tokenizer, base, assembly) workload = do
    tasks' <- traverse prepare (Workload.tasks workload)
    first show (C.withCohort (C.Definition policy' tasks') (const ()))
    pure (Cycle tasks' (Workload.order workload) (Workload.delivery workload))
  where
    prepare sample = do
        planned <- first show (I.prepare I.Request {I.artifact = policy', I.tokenizer = tokenizer, I.base = base, I.assembly = assembly, I.prompt = Workload.prompt sample, I.tokens = Workload.tokens sample, I.temperature = Workload.temperature sample, I.seed = Workload.seed sample})
        pure (C.Task (Workload.name sample) (Workload.group sample) planned (Workload.rule sample))

scoring :: Config -> Checkpoint -> Maybe Batch.Reference
scoring config selected
    | L.reference (settings config) == policy selected = Nothing
    | otherwise = Just (Batch.Reference (reference config) (L.reference (settings config)))

inferenceWorker :: Config -> Checkpoint -> Infer.Worker
inferenceWorker config selected =
    Infer.Worker (inferencePython engine) (inference engine) (cache engine) (directory selected </> "adapter.safetensors") [] (inferenceConfiguration engine)
  where
    engine = backend config

learnerSettings :: Config -> Cursor -> L.Settings
learnerSettings config (Cursor index selected _) = (settings config) {L.policy = policy selected, L.learner = learner selected, L.schedule = L.synchronous index (policy selected)}

execute :: Driver scope -> (Cursor, L.Plan scope) -> (forall value. IO value -> IO value) -> IO (Either Error (Generation scope))
execute driver (cursor, planned) restore = do
    ordinal <- Internal.reserve (rollout driver) oneUpdate
    let binding = V.ordinal ordinal
    case W.prepare binding planned of
        Left problem -> reset driver cursor >> pure (Left (Update problem))
        Right call -> do
            writeIORef (state driver) (Updating cursor binding)
            let Cursor _ selected _ = cursor
                worker = updateWorker (configuration driver) (selected, ordinal)
            executed <- restore (Owner.run (updater driver) (Resident.Paths (W.checkpoint worker) (W.output worker)) call)
            case executed of
                Left problem -> pure (Left (Update problem))
                Right completed -> Right <$> publish driver (cursor, completed)
  where
    oneUpdate = 1

updateWorker :: Config -> (Checkpoint, Natural) -> W.Worker
updateWorker config (selected, ordinal) =
    W.Worker
        { W.executable = python (backend config)
        , W.script = learning (backend config)
        , W.cache = cache (backend config)
        , W.checkpoint = directory selected
        , W.reference = reference config
        , W.output = root config </> stagedName ordinal
        }

publish :: Driver scope -> (Cursor, Observation.Observation scope) -> IO (Generation scope)
publish driver (cursor@(Cursor previous _ description), executed) = do
    writeIORef (state driver) (Publishing cursor executed)
    let config = configuration driver
        produced = Observation.report executed
        V.CallId ordinal = V.boundCall (V.completedBinding (P.completion produced))
        next = previous + 1
        name = "generation" ++ show next
        location = S.Location (root config) (Bytes.pack (stagedName ordinal)) (Bytes.pack name)
    updated <- either (ioError . userError) pure (Policy.successor (P.adapter produced) description)
    Policy.stageDescription (root config </> stagedName ordinal </> "policy.json") updated
    committed <- S.publishCheckpoint (publication config) location
    let selected = Checkpoint (root config </> name) (P.adapter produced) (P.learner produced)
    writeIORef (state driver) (Idle (Cursor next selected updated))
    pure (Generation selected executed committed)

stagedName :: Natural -> String
stagedName ordinal = "staging" ++ show ordinal

reset :: Driver scope -> Cursor -> IO ()
reset driver cursor = writeIORef (state driver) (Idle cursor)

describe :: State scope -> Status
describe (Idle (Cursor index selected _)) = Ready index selected
describe (Collecting (Cursor index _ _)) = Rolling index
describe (Updating (Cursor index _ _) binding) = Unresolved index binding
describe (Publishing (Cursor index _ _) executed) = Committing index (V.completedBinding (P.completion (Observation.report executed)))

status :: Driver scope -> IO Status
status = fmap describe . readIORef . state

current :: Generation scope -> Checkpoint
current (Generation value _ _) = value

result :: Generation scope -> P.Result
result (Generation _ executed _) = Observation.report executed

plan :: Generation scope -> L.Plan scope
plan (Generation _ executed _) = Observation.plan executed

receipt :: Generation scope -> S.Receipt
receipt (Generation _ _ committed) = committed
