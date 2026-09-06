{-# LANGUAGE Safe #-}

module Invar.Spec.Request (
    Core,
    mkCore,
    payload,
    prompt,
    limit,
    Semantics (..),
    RequestId (..),
    Status (..),
    Request (..),
    Config,
    Action (..),
    Event (..),
    Invalid (..),
    Guard (..),
    Reason (..),
    Outcome (..),
    initialize,
    step,
    stopped,
    context,
    extent,
    unroll,
) where

import Control.Monad (foldM)
import Data.List.NonEmpty (NonEmpty)
import Data.List.NonEmpty qualified as NonEmpty
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Numeric.Natural (Natural)

data Core input token = Core input (NonEmpty token) Natural
    deriving (Eq, Show)

mkCore :: input -> NonEmpty token -> Natural -> Either Invalid (Core input token)
mkCore _ _ 0 = Left ZeroLimit
mkCore input tokens cap = Right (Core input tokens cap)

payload :: Core input token -> input
payload (Core input _ _) = input

prompt :: Core input token -> NonEmpty token
prompt (Core _ tokens _) = tokens

limit :: Core input token -> Natural
limit (Core _ _ cap) = cap

data Semantics input token = Semantics
    { next :: Core input token -> [token] -> token
    , frontier :: Core input token -> Natural -> Natural
    , terminal :: token -> Bool
    }

newtype RequestId = RequestId Natural
    deriving (Eq, Ord, Show)

data Status = Pool | Active | Paused | Done | Cancelled
    deriving (Eq, Show)

data Request input token = Request
    { core :: Core input token
    , status :: Status
    , cached :: Natural
    , generated :: [token]
    }
    deriving (Eq, Show)

type Config input token = Map RequestId (Request input token)

data Action = Admit | Chunk | Decode | Preempt Natural | Resume | Cancel | Complete
    deriving (Eq, Show)

data Event = Event {requestId :: RequestId, action :: Action}
    deriving (Eq, Show)

data Invalid = ZeroLimit | DuplicateRequest RequestId
    deriving (Eq, Show)

data Guard
    = CacheNotBehind
    | CacheNotReady
    | AlreadyStopped
    | NotStopped
    | CacheIncrease
    | NonAdvancingFrontier
    deriving (Eq, Show)

data Reason
    = UnknownRequest RequestId
    | StatusMismatch RequestId [Status] Status
    | GuardFailed RequestId Guard
    deriving (Eq, Show)

data Outcome input token = Applied (Config input token) | Rejected Reason
    deriving (Eq, Show)

initialize :: [(RequestId, Core input token)] -> Either Invalid (Config input token)
initialize = foldM insert Map.empty
  where
    insert config (rid, input) =
        if Map.member rid config
            then Left (DuplicateRequest rid)
            else Right (Map.insert rid (Request input Pool 0 []) config)

step :: Semantics input token -> Config input token -> Event -> Outcome input token
step semantics config (Event rid event) = case Map.lookup rid config of
    Nothing -> Rejected (UnknownRequest rid)
    Just request
        | status request `notElem` required event ->
            Rejected (StatusMismatch rid (required event) (status request))
        | otherwise -> case apply semantics request event of
            Left reason -> Rejected (GuardFailed rid reason)
            Right result -> Applied (Map.insert rid result config)

required :: Action -> [Status]
required Admit = [Pool]
required Resume = [Paused]
required Cancel = [Active, Paused]
required _ = [Active]

apply :: Semantics input token -> Request input token -> Action -> Either Guard (Request input token)
apply semantics request event = case event of
    Admit -> Right request {status = Active, cached = 0, generated = []}
    Chunk -> chunk semantics request
    Decode -> decode semantics request
    Preempt position
        | position > cached request -> Left CacheIncrease
        | otherwise -> Right request {status = Paused, cached = position}
    Resume -> Right request {status = Active}
    Cancel -> Right request {status = Cancelled}
    Complete
        | stopped semantics request -> Right request {status = Done}
        | otherwise -> Left NotStopped

chunk :: Semantics input token -> Request input token -> Either Guard (Request input token)
chunk semantics request
    | cached request >= extent request = Left CacheNotBehind
    | position <= cached request = Left NonAdvancingFrontier
    | otherwise = Right request {cached = min position (extent request)}
  where
    position = frontier semantics (core request) (cached request)

decode :: Semantics input token -> Request input token -> Either Guard (Request input token)
decode semantics request
    | cached request /= extent request = Left CacheNotReady
    | stopped semantics request = Left AlreadyStopped
    | otherwise =
        Right
            request
                { cached = cached request + 1
                , generated = generated request ++ [next semantics (core request) (context request)]
                }

stopped :: Semantics input token -> Request input token -> Bool
stopped semantics request = fromIntegral (length tokens) == limit (core request) || ends tokens
  where
    tokens = generated request
    ends [] = False
    ends xs = terminal semantics (last xs)

context :: Request input token -> [token]
context request = NonEmpty.toList (prompt (core request)) ++ generated request

extent :: Request input token -> Natural
extent = fromIntegral . length . context

unroll :: Semantics input token -> Core input token -> [token]
unroll semantics input = go []
  where
    go tokens
        | stopped semantics request = tokens
        | otherwise = go (tokens ++ [next semantics input (context request)])
      where
        request = Request input Active 0 tokens
