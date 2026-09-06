module Model (
    State,
    Settings (..),
    Mutation (..),
    Experiment (Experiment),
    mutation,
    TraceError (..),
    tableWidth,
    semantics,
    oracle,
    runStep,
    execute,
    require,
    requireLookup,
    target,
    other,
    completion,
) where

import Data.Bits (testBit)
import Data.Map.Strict qualified as Map
import Invar.Spec.Request
import Numeric.Natural (Natural)

type State = Config Int Bool
type Step = State -> Event -> Outcome Int Bool

data Settings = Settings {table :: Int, stopOnTrue :: Bool, stride :: Natural}
    deriving (Eq, Show)

data Mutation = Unchanged | NoStoppedGuard | LeakyDecode | TruncatingPreempt Natural
    deriving (Eq, Show)

data Experiment = Experiment {settings :: Settings, mutation :: Mutation}
    deriving (Eq, Show)

data TraceError = TraceError Int Event Reason
    deriving (Eq, Show)

tableWidth :: Int
tableWidth = 8

semantics :: Settings -> Semantics Int Bool
semantics options =
    Semantics
        { next = oracle options
        , frontier = \_ position -> position + stride options
        , terminal = \token -> stopOnTrue options && token
        }

oracle :: Settings -> Core Int Bool -> [Bool] -> Bool
oracle options input tokens = testBit (table options) index
  where
    index = (payload input + foldl' (\n bit -> 2 * n + fromEnum bit) 0 tokens) `mod` tableWidth

runStep :: Experiment -> Step
runStep experiment config event = case mutation experiment of
    Unchanged -> step meaning config event
    NoStoppedGuard -> withoutStopped meaning config event
    LeakyDecode -> step meaning {next = \_ _ -> activeCount /= 1} config event
    TruncatingPreempt keep -> truncateAfter keep (step meaning config event) event
  where
    meaning = semantics (settings experiment)
    activeCount = Map.size (Map.filter ((== Active) . status) config)

withoutStopped :: Semantics Int Bool -> Step
withoutStopped meaning config (Event rid Decode)
    | Just request <- Map.lookup rid config
    , status request == Active
    , cached request == extent request =
        Applied
            ( Map.insert
                rid
                request
                    { cached = cached request + 1
                    , generated = generated request ++ [next meaning (core request) (context request)]
                    }
                config
            )
withoutStopped meaning config event = step meaning config event

truncateAfter :: Natural -> Outcome Int Bool -> Event -> Outcome Int Bool
truncateAfter keep (Applied config) (Event rid (Preempt _)) =
    Applied (Map.adjust (\request -> request {generated = take (fromIntegral keep) (generated request)}) rid config)
truncateAfter _ outcome _ = outcome

execute :: (Config input token -> Event -> Outcome input token) -> Config input token -> [Event] -> Either TraceError [Config input token]
execute transition = go 0
  where
    go _ config [] = Right [config]
    go index config (event : rest) = case transition config event of
        Rejected reason -> Left (TraceError index event reason)
        Applied result -> (config :) <$> go (index + 1) result rest

require :: (Show problem) => Either problem value -> value
require = either (error . show) id

requireLookup :: RequestId -> Config input token -> Request input token
requireLookup rid config = case Map.lookup rid config of
    Nothing -> error ("Missing fixture request: " ++ show rid)
    Just request -> request

target, other :: RequestId
target = RequestId 0
other = RequestId 1

progressAction :: Semantics input token -> Request input token -> Either String (Maybe Action)
progressAction meaning request = case status request of
    Pool -> Right (Just Admit)
    Paused -> Right (Just Resume)
    Done -> Right Nothing
    Cancelled -> Left "Cannot complete a cancelled request"
    Active -> activeAction meaning request

activeAction :: Semantics input token -> Request input token -> Either String (Maybe Action)
activeAction meaning request
    | stopped meaning request = Right (Just Complete)
    | cached request < extent request = Right (Just Chunk)
    | cached request == extent request = Right (Just Decode)
    | otherwise = Left "Cache ahead of context: no forward-progress action"

completion :: Settings -> State -> RequestId -> Either String [Event]
completion options initial rid = go initial
  where
    meaning = semantics options
    go config = do
        candidate <- progressAction meaning (requireLookup rid config)
        case candidate of
            Nothing -> Right []
            Just event -> case step meaning config (Event rid event) of
                Rejected reason -> Left (show reason)
                Applied result -> (Event rid event :) <$> go result
