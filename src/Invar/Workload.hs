{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedStrings #-}

module Invar.Workload (DocumentOf, Document, CycleOf, Cycle, TaskOf, Task, Declared (..), Shape (..), decode, decodeOf, digest, value, describe, shape, cycles, tasks, order, delivery, name, group, prompt, tokens, temperature, seed, rule, grade) where

import Control.Monad (unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Key, Object, Value, object, parseJSON, withObject, (.:), (.=))
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Invar.Artifact qualified as Artifact
import Invar.Cohort qualified as Cohort
import Invar.Json qualified as Json
import Invar.Reward qualified as Reward
import Invar.Schedule qualified as Schedule
import Numeric.Natural (Natural)

data DocumentOf grade = Document String Value [CycleOf grade]
    deriving (Eq, Show)

type Document = DocumentOf Cohort.Graded

data CycleOf grade = Cycle [TaskOf grade] [Natural] [Natural]
    deriving (Eq, Show)

type Cycle = CycleOf Cohort.Graded

data TaskOf grade = Task
    { taskName :: String
    , taskPrompt :: String
    , taskTokens :: Natural
    , taskTemperature :: Double
    , taskSeed :: Integer
    , taskGrade :: grade
    , taskValue :: Value
    }
    deriving (Eq, Show)

type Task = TaskOf Cohort.Graded

class (Cohort.Grade grade) => Declared grade where
    declaredFields :: [Key]
    declared :: Object -> Parser grade

instance Declared Cohort.Graded where
    declaredFields = ["group", "answer"]
    declared fields = do
        expected <- fields .: "answer"
        selected <- either (fail . show) pure (Reward.decimal expected)
        Cohort.Graded <$> fields .: "group" <*> pure selected

instance Declared Cohort.Ungraded where
    declaredFields = []
    declared _ = pure Cohort.Ungraded

decode :: ByteString -> Either String Document
decode = decodeOf

decodeOf :: (Declared grade) => ByteString -> Either String (DocumentOf grade)
decodeOf encoded = do
    original <- Json.decode encoded
    declaredCycles <- parseEither parseJSON original
    when (null declaredCycles) (Left "Expected nonempty workload cohorts")
    checked <- traverse (parseEither parseCycle) declaredCycles
    pure (Document (Artifact.hex (SHA256.hash encoded)) original checked)

parseCycle :: (Declared grade) => Value -> Parser (CycleOf grade)
parseCycle = withObject "Workload cohort" $ \fields -> do
    Json.fields ["tasks", "order", "delivery"] fields
    chosen <- fields .: "tasks" >>= traverse parseTask
    execution <- fields .: "order"
    arrival <- fields .: "delivery"
    either (fail . show) pure (Cohort.validateMembers [(name task, grade task) | task <- chosen])
    _ <- either (fail . show) pure (Schedule.prepare (fromIntegral (length chosen)) execution arrival)
    pure (Cycle chosen execution arrival)

parseTask :: forall grade. (Declared grade) => Value -> Parser (TaskOf grade)
parseTask original = withObject "Workload task" parse original
  where
    parse fields = do
        Json.fields (["name", "prompt", "tokens", "temperature", "seed"] ++ declaredFields @grade) fields
        count <- fields .: "tokens"
        thermal <- fields .: "temperature" >>= Json.finite
        input <- fields .: "prompt"
        unless (count > 0 && thermal > 0) (fail "Workload token budget and temperature must be positive")
        when ('\0' `elem` input) (fail "A process argument cannot contain NUL")
        Task <$> fields .: "name" <*> pure input <*> pure count <*> pure thermal <*> fields .: "seed" <*> declared fields <*> pure original

digest :: DocumentOf grade -> String
digest (Document identity _ _) = identity

value :: DocumentOf grade -> Value
value (Document _ original _) = original

describe :: DocumentOf grade -> Value
describe document = object ["digest" .= digest document, "cohorts" .= shapeCohorts (shape document)]

data Shape = Shape {shapeDigest :: String, shapeCohorts :: [[Value]], shapeCycles :: [([String], [Natural], [Natural])]}
    deriving (Eq, Show)

shape :: DocumentOf grade -> Shape
shape document = Shape (digest document) [map taskValue (tasks selected) | selected <- cycles document] [(map name (tasks selected), order selected, delivery selected) | selected <- cycles document]

cycles :: DocumentOf grade -> [CycleOf grade]
cycles (Document _ _ listed) = listed

tasks :: CycleOf grade -> [TaskOf grade]
tasks (Cycle listed _ _) = listed

order :: CycleOf grade -> [Natural]
order (Cycle _ execution _) = execution

delivery :: CycleOf grade -> [Natural]
delivery (Cycle _ _ arrival) = arrival

name :: TaskOf grade -> String
name = taskName

group :: Task -> String
group task = let Cohort.Graded membership _ = taskGrade task in membership

prompt :: TaskOf grade -> String
prompt = taskPrompt

tokens :: TaskOf grade -> Natural
tokens = taskTokens

temperature :: TaskOf grade -> Double
temperature = taskTemperature

seed :: TaskOf grade -> Integer
seed = taskSeed

rule :: Task -> Reward.Rule
rule task = let Cohort.Graded _ chosen = taskGrade task in chosen

grade :: TaskOf grade -> grade
grade = taskGrade
