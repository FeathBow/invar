{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

module Invar.Learn.Program (checked, schema, distilled, distilledSchema) where

import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Word (Word32)
import Invar.Construct qualified as C
import Invar.Learn.Algorithm qualified as Algorithm
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Program qualified as P
import Numeric.Natural (Natural)

type Optimizer = C.Record '[ '("learning_rate", Rational), '("betas", [Rational]), '("epsilon", Rational), '("weight_decay", Rational)]
type Learner = C.Record '[ '("policy", [Natural]), '("learner", [Natural]), '("tokenizer", [Natural]), '("base", [Natural]), '("assembly", [Natural]), '("optimizer", Optimizer)]
type Algorithm = C.Record '[ '("epsilon", Rational), '("penalty", Rational), '("delta", Rational), '("steps", Rational)]
type Trajectory = C.Record '[ '("prompt", [Natural]), '("seed", Rational), '("limit", Natural), '("temperature", Rational), '("tokens", [Natural]), '("prompt_length", Natural), '("text", [Natural]), '("truncated", Bool)]
type Policy = C.Record '[ '("artifact", [Natural]), '("profile", [Natural])]
type BehaviorModel = C.Record '[ '("base", [Natural]), '("assembly", [Natural])]
type Schedule = C.Record '[ '("update", Natural), '("staleness", Natural)]
type Generation = C.Record '[ '("version", Natural), '("policy", [Natural])]
type Teacher = C.Record '[ '("model", [Natural]), '("revision", [Natural]), '("adapter", [Natural]), '("tokenizer", [Natural]), '("base", [Natural]), '("assembly", [Natural]), '("temperature", Rational), '("truncation", [Natural]), '("logprobs", [Natural]), '("path", [Natural])]
type Binding = C.Record '[ '("call", Natural), '("attempt", Natural), '("instance", Natural)]
type Coefficients = C.Record '[ '("epsilon", Rational), '("penalty", Rational), '("steps", Rational)]
type Distilled = '[ 'C.Semantic "policy", 'C.Semantic "learner", 'C.Semantic "reference", 'C.Semantic "reference_source", 'C.Semantic "algorithm", 'C.Semantic "trajectories", 'C.Semantic "behavior_model", 'C.Semantic "schedule", 'C.Semantic "generations", 'C.Semantic "behavior", 'C.Semantic "reference_scores", 'C.Semantic "teacher", 'C.Semantic "teacher_scores", 'C.Semantic "teacher_calls", 'C.Semantic "order"]
type Allowed = '[ 'C.Semantic "policy", 'C.Semantic "learner", 'C.Semantic "reference", 'C.Semantic "reference_source", 'C.Semantic "algorithm", 'C.Semantic "trajectories", 'C.Semantic "behavior_model", 'C.Semantic "schedule", 'C.Semantic "generations", 'C.Semantic "behavior", 'C.Semantic "reference_scores", 'C.Semantic "rewards", 'C.Semantic "groups", 'C.Semantic "order"]

checked :: Either C.BuildError A.Checked
checked = C.compile (E.Semantics schema Map.empty) [C.emit @Algorithm.Sink @Algorithm.Grpo @Allowed expression]
  where
    expression =
        C.record
            $ C.field @"policy" (C.source @('C.Semantic "policy") @Policy)
            $ C.field @"learner" (C.source @('C.Semantic "learner") @Learner)
            $ C.field @"reference" (C.sequenceSource @('C.Semantic "reference") @Natural)
            $ C.field @"algorithm" (C.source @('C.Semantic "algorithm") @Algorithm)
            $ C.field @"trajectories" (C.mapSource @('C.Semantic "trajectories") @Trajectory)
            $ C.field @"behavior_model" (C.source @('C.Semantic "behavior_model") @BehaviorModel)
            $ C.field @"schedule" (C.source @('C.Semantic "schedule") @Schedule)
            $ C.field @"reference_source" (C.sequenceSource @('C.Semantic "reference_source") @Natural)
            $ C.field @"generations" (C.mapSource @('C.Semantic "generations") @Generation)
            $ C.field @"behavior" (C.mapSource @('C.Semantic "behavior") @[Word32])
            $ C.field @"reference_scores" (C.mapSource @('C.Semantic "reference_scores") @[Word32])
            $ C.field @"rewards" (C.mapSource @('C.Semantic "rewards") @Rational)
            $ C.field @"groups" (C.sequenceSource @('C.Semantic "groups") @(Map Natural Bool))
            $ C.field @"order" (C.sequenceSource @('C.Semantic "order") @(Map Natural Bool)) C.emptyFields

distilled :: Either C.BuildError A.Checked
distilled = C.compile (E.Semantics distilledSchema Map.empty) [C.emit @Algorithm.Sink @Algorithm.Opd @Distilled expression]
  where
    expression =
        C.record
            $ C.field @"policy" (C.source @('C.Semantic "policy") @Policy)
            $ C.field @"learner" (C.source @('C.Semantic "learner") @Learner)
            $ C.field @"reference" (C.sequenceSource @('C.Semantic "reference") @Natural)
            $ C.field @"algorithm" (C.source @('C.Semantic "algorithm") @Coefficients)
            $ C.field @"trajectories" (C.mapSource @('C.Semantic "trajectories") @Trajectory)
            $ C.field @"behavior_model" (C.source @('C.Semantic "behavior_model") @BehaviorModel)
            $ C.field @"schedule" (C.source @('C.Semantic "schedule") @Schedule)
            $ C.field @"reference_source" (C.sequenceSource @('C.Semantic "reference_source") @Natural)
            $ C.field @"generations" (C.mapSource @('C.Semantic "generations") @Generation)
            $ C.field @"behavior" (C.mapSource @('C.Semantic "behavior") @[Word32])
            $ C.field @"reference_scores" (C.mapSource @('C.Semantic "reference_scores") @[Word32])
            $ C.field @"teacher" (C.source @('C.Semantic "teacher") @Teacher)
            $ C.field @"teacher_scores" (C.mapSource @('C.Semantic "teacher_scores") @[Word32])
            $ C.field @"teacher_calls" (C.mapSource @('C.Semantic "teacher_calls") @Binding)
            $ C.field @"order" (C.sequenceSource @('C.Semantic "order") @(Map Natural Bool)) C.emptyFields

schema :: P.Schema
schema = declared Algorithm.grpo (Map.union shared (Map.fromList [("algorithm", record [("epsilon", P.NumberType), ("penalty", P.NumberType), ("delta", P.NumberType), ("steps", P.NumberType)]), ("rewards", P.MapType P.NumberType), ("groups", selectors)]))

distilledSchema :: P.Schema
distilledSchema = declared Algorithm.opd (Map.union shared (Map.fromList [("algorithm", record [("epsilon", P.NumberType), ("penalty", P.NumberType), ("steps", P.NumberType)]), ("teacher", teacher), ("teacher_scores", P.MapType (P.SequenceType P.BitsType)), ("teacher_calls", P.MapType (record [("call", P.TokenType), ("attempt", P.TokenType), ("instance", P.TokenType)]))]))
  where
    teacher = record ([(name, text) | name <- ["model", "revision", "adapter", "tokenizer", "base", "assembly", "truncation", "logprobs", "path"]] ++ [("temperature", P.NumberType)])

declared :: String -> Map String P.Type -> P.Schema
declared specification fields = P.Schema sources Map.empty (Map.singleton Algorithm.sink (P.Sink specification (P.RecordType fields) (Map.keysSet sources) Set.empty))
  where
    sources = Map.mapKeys P.Semantic fields

shared :: Map String P.Type
shared = Map.fromList [("policy", record [("artifact", text), ("profile", text)]), ("learner", learner), ("reference", text), ("reference_source", text), ("trajectories", P.MapType trajectory), ("behavior_model", record [("base", text), ("assembly", text)]), ("schedule", record [("update", P.TokenType), ("staleness", P.TokenType)]), ("generations", P.MapType (record [("version", P.TokenType), ("policy", text)])), ("behavior", P.MapType (P.SequenceType P.BitsType)), ("reference_scores", P.MapType (P.SequenceType P.BitsType)), ("order", selectors)]
  where
    optimizer = record [("learning_rate", P.NumberType), ("betas", P.SequenceType P.NumberType), ("epsilon", P.NumberType), ("weight_decay", P.NumberType)]
    learner = record [("policy", text), ("learner", text), ("tokenizer", text), ("base", text), ("assembly", text), ("optimizer", optimizer)]
    trajectory = record [("prompt", text), ("seed", P.NumberType), ("limit", P.TokenType), ("temperature", P.NumberType), ("tokens", P.SequenceType P.TokenType), ("prompt_length", P.TokenType), ("text", text), ("truncated", P.BooleanType)]

selectors :: P.Type
selectors = P.SequenceType (P.MapType P.BooleanType)

text :: P.Type
text = P.SequenceType P.TokenType

record :: [(String, P.Type)] -> P.Type
record = P.RecordType . Map.fromList
