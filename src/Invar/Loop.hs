{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RoleAnnotations #-}

module Invar.Loop (Backend (..), Config (..), Cycle (..), Checkpoint (..), Driver, Generation, Status (..), Error (..), withDriver, run, status, current, result, receipt) where

import Control.Concurrent.MVar (MVar, newMVar, withMVar)
import Control.Exception (mask, onException)
import Data.Bifunctor (first)
import Data.ByteString.Char8 qualified as Bytes
import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import Invar.Cohort qualified as C
import Invar.Infer qualified as I
import Invar.Learn qualified as L
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Worker qualified as W
import Invar.Rollout qualified as R
import Invar.Rollout.Internal qualified as Internal
import Invar.Spec.Invocation qualified as V
import Invar.Store qualified as S
import Invar.Worker qualified as Infer
import Numeric.Natural (Natural)
import System.Directory (createDirectory)
import System.FilePath ((</>))

data Backend = Backend {python :: FilePath, inference :: FilePath, learning :: FilePath, cache :: FilePath, sessions :: [[(String, String)]]}
data Config = Config {backend :: Backend, root :: FilePath, checkpoint :: FilePath, reference :: FilePath, settings :: L.Settings, publication :: S.Method}
data Cycle = Cycle {tasks :: [C.Task], order :: [Natural], delivery :: [Natural]}
data Checkpoint = Checkpoint {directory :: FilePath, policy :: String, learner :: String}
    deriving (Eq, Show)

type role Driver nominal
data Driver scope = Driver {configuration :: Config, rollout :: R.Driver scope, lock :: MVar (), state :: IORef (State scope)}

type role Generation nominal
data Generation scope = Generation Checkpoint (W.Execution scope) S.Receipt

data Cursor = Cursor Natural Checkpoint
data State scope = Idle Cursor | Collecting Cursor | Updating Cursor V.Binding | Publishing Cursor (W.Execution scope)

data Status = Ready Natural Checkpoint | Rolling Natural | Unresolved Natural V.Binding | Committing Natural V.Binding
    deriving (Eq, Show)

data Error = Settings L.Error | NotReady Status | Rollout R.Error | Plan L.Error | Update W.Failure
    deriving (Eq, Show)

withDriver :: Config -> (forall scope. Driver scope -> IO value) -> IO (Either Error value)
withDriver config action = case L.validate (settings config) of
    Left problem -> pure (Left (Settings problem))
    Right () -> do
        createDirectory (root config)
        R.withDriver $ \collector -> do
            gate <- newMVar ()
            position <- newIORef (Idle (Cursor initialGeneration initial))
            Right <$> action (Driver config collector gate position)
  where
    initialGeneration = 0
    initial = Checkpoint (checkpoint config) (L.policy (settings config)) (L.learner (settings config))

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
collect driver (cursor@(Cursor _ selected), workload) =
    case mapM_ (L.materialization (settings config) . I.requested . C.plan) (tasks workload) of
        Left problem -> pure (Left (Plan problem))
        Right () -> do
            let engine = backend config
                worker = Infer.Worker (python engine) (inference engine) (cache engine) (directory selected </> "adapter.safetensors") []
                options = R.Options {R.worker = worker, R.sessions = sessions engine, R.definition = C.Definition (policy selected) (tasks workload), R.order = order workload, R.delivery = delivery workload}
            generated <- R.run (rollout driver) options
            pure $ do
                batch <- first Rollout generated
                first Plan (L.prepare (learnerSettings config cursor) batch)
  where
    config = configuration driver

learnerSettings :: Config -> Cursor -> L.Settings
learnerSettings config (Cursor _ selected) = (settings config) {L.policy = policy selected, L.learner = learner selected}

execute :: Driver scope -> (Cursor, L.Plan scope) -> (forall value. IO value -> IO value) -> IO (Either Error (Generation scope))
execute driver (cursor, planned) restore = do
    ordinal <- Internal.reserve (rollout driver) oneUpdate
    let binding = V.Binding (V.CallId ordinal) (V.AttemptId ordinal) (V.Instance ordinal)
    case W.prepare binding planned of
        Left problem -> reset driver cursor >> pure (Left (Update problem))
        Right call -> do
            writeIORef (state driver) (Updating cursor binding)
            executed <- restore (W.run (updateWorker (configuration driver) (cursor, ordinal)) call)
            case executed of
                Left problem -> pure (Left (Update problem))
                Right completed -> Right <$> publish driver (cursor, completed)
  where
    oneUpdate = 1

updateWorker :: Config -> (Cursor, Natural) -> W.Worker
updateWorker config (Cursor _ selected, ordinal) =
    W.Worker
        { W.executable = python (backend config)
        , W.script = learning (backend config)
        , W.cache = cache (backend config)
        , W.checkpoint = directory selected
        , W.reference = reference config
        , W.output = root config </> stagedName ordinal
        }

publish :: Driver scope -> (Cursor, W.Execution scope) -> IO (Generation scope)
publish driver (cursor@(Cursor previous _), executed) = do
    writeIORef (state driver) (Publishing cursor executed)
    let config = configuration driver
        produced = W.report executed
        V.CallId ordinal = V.boundCall (V.completedBinding (P.completion produced))
        next = previous + 1
        name = "generation" ++ show next
        location = S.Location (root config) (Bytes.pack (stagedName ordinal)) (Bytes.pack name)
    committed <- S.publishCheckpoint (publication config) location
    let selected = Checkpoint (root config </> name) (P.adapter produced) (P.learner produced)
    writeIORef (state driver) (Idle (Cursor next selected))
    pure (Generation selected executed committed)

stagedName :: Natural -> String
stagedName ordinal = "staging" ++ show ordinal

reset :: Driver scope -> Cursor -> IO ()
reset driver cursor = writeIORef (state driver) (Idle cursor)

describe :: State scope -> Status
describe (Idle (Cursor index selected)) = Ready index selected
describe (Collecting (Cursor index _)) = Rolling index
describe (Updating (Cursor index _) binding) = Unresolved index binding
describe (Publishing (Cursor index _) executed) = Committing index (V.completedBinding (P.completion (W.report executed)))

status :: Driver scope -> IO Status
status = fmap describe . readIORef . state

current :: Generation scope -> Checkpoint
current (Generation value _ _) = value

result :: Generation scope -> W.Execution scope
result (Generation _ executed _) = executed

receipt :: Generation scope -> S.Receipt
receipt (Generation _ _ committed) = committed
