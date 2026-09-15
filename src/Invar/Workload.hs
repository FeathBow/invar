{-# LANGUAGE OverloadedStrings #-}

module Invar.Workload (Document, Cycle, Task, decode, digest, value, describe, cycles, tasks, order, delivery, name, group, prompt, tokens, temperature, seed, rule) where

import Control.Monad (unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value, object, parseJSON, withObject, (.:), (.=))
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Invar.Artifact qualified as Artifact
import Invar.Cohort qualified as Cohort
import Invar.Json qualified as Json
import Invar.Reward qualified as Reward
import Invar.Schedule qualified as Schedule
import Numeric.Natural (Natural)

data Document = Document String Value [Cycle]
    deriving (Eq, Show)

data Cycle = Cycle [Task] [Natural] [Natural]
    deriving (Eq, Show)

data Task = Task
    { taskName :: String
    , taskGroup :: String
    , taskPrompt :: String
    , taskTokens :: Natural
    , taskTemperature :: Double
    , taskSeed :: Integer
    , taskRule :: Reward.Rule
    , taskValue :: Value
    }
    deriving (Eq, Show)

decode :: ByteString -> Either String Document
decode encoded = do
    original <- Json.decode encoded
    declared <- parseEither parseJSON original
    when (null declared) (Left "Expected nonempty workload cohorts")
    checked <- traverse (parseEither parseCycle) declared
    pure (Document (Artifact.hex (SHA256.hash encoded)) original checked)

parseCycle :: Value -> Parser Cycle
parseCycle = withObject "Workload cohort" $ \fields -> do
    Json.fields ["tasks", "order", "delivery"] fields
    declared <- fields .: "tasks" >>= traverse parseTask
    execution <- fields .: "order"
    arrival <- fields .: "delivery"
    either (fail . show) pure (Cohort.validateMembers [(name task, group task) | task <- declared])
    _ <- either (fail . show) pure (Schedule.prepare (fromIntegral (length declared)) execution arrival)
    pure (Cycle declared execution arrival)

parseTask :: Value -> Parser Task
parseTask original = withObject "Workload task" parse original
  where
    parse fields = do
        Json.fields ["name", "group", "prompt", "tokens", "temperature", "seed", "answer"] fields
        expected <- fields .: "answer"
        selected <- either (fail . show) pure (Reward.decimal expected)
        count <- fields .: "tokens"
        thermal <- fields .: "temperature" >>= Json.finite
        input <- fields .: "prompt"
        unless (count > 0 && thermal > 0) (fail "Workload token budget and temperature must be positive")
        when ('\0' `elem` input) (fail "A process argument cannot contain NUL")
        Task <$> fields .: "name" <*> fields .: "group" <*> pure input <*> pure count <*> pure thermal <*> fields .: "seed" <*> pure selected <*> pure original

digest :: Document -> String
digest (Document identity _ _) = identity

value :: Document -> Value
value (Document _ original _) = original

describe :: Document -> Value
describe document = object ["digest" .= digest document, "cohorts" .= map (map taskValue . tasks) (cycles document)]

cycles :: Document -> [Cycle]
cycles (Document _ _ declared) = declared

tasks :: Cycle -> [Task]
tasks (Cycle declared _ _) = declared

order :: Cycle -> [Natural]
order (Cycle _ execution _) = execution

delivery :: Cycle -> [Natural]
delivery (Cycle _ _ arrival) = arrival

name :: Task -> String
name = taskName

group :: Task -> String
group = taskGroup

prompt :: Task -> String
prompt = taskPrompt

tokens :: Task -> Natural
tokens = taskTokens

temperature :: Task -> Double
temperature = taskTemperature

seed :: Task -> Integer
seed = taskSeed

rule :: Task -> Reward.Rule
rule = taskRule
