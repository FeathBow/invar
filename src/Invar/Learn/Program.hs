{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

module Invar.Learn.Program (checked, schema) where

import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Word (Word32)
import Invar.Construct qualified as C
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
type Allowed = '[ 'C.Semantic "policy", 'C.Semantic "learner", 'C.Semantic "reference", 'C.Semantic "algorithm", 'C.Semantic "trajectories", 'C.Semantic "behavior_model", 'C.Semantic "behavior", 'C.Semantic "reference_scores", 'C.Semantic "rewards", 'C.Semantic "groups", 'C.Semantic "order"]

checked :: Either C.BuildError A.Checked
checked = C.compile (E.Semantics schema Map.empty) [C.emit @"update" @"grpo-token-mean/v1" @Allowed expression]
  where
    expression =
        C.record
            $ C.field @"policy" (C.source @('C.Semantic "policy") @Policy)
            $ C.field @"learner" (C.source @('C.Semantic "learner") @Learner)
            $ C.field @"reference" (C.sequenceSource @('C.Semantic "reference") @Natural)
            $ C.field @"algorithm" (C.source @('C.Semantic "algorithm") @Algorithm)
            $ C.field @"trajectories" (C.mapSource @('C.Semantic "trajectories") @Trajectory)
            $ C.field @"behavior_model" (C.source @('C.Semantic "behavior_model") @BehaviorModel)
            $ C.field @"behavior" (C.mapSource @('C.Semantic "behavior") @[Word32])
            $ C.field @"reference_scores" (C.mapSource @('C.Semantic "reference_scores") @[Word32])
            $ C.field @"rewards" (C.mapSource @('C.Semantic "rewards") @Rational)
            $ C.field @"groups" (C.sequenceSource @('C.Semantic "groups") @(Map Natural Bool))
            $ C.field @"order" (C.sequenceSource @('C.Semantic "order") @(Map Natural Bool)) C.emptyFields

schema :: P.Schema
schema = P.Schema sources Map.empty (Map.singleton "update" sink)
  where
    sources = Map.mapKeys P.Semantic fields
    sink = P.Sink "grpo-token-mean/v1" (P.RecordType fields) (Map.keysSet sources) Set.empty

fields :: Map String P.Type
fields = Map.fromList [("policy", record [("artifact", text), ("profile", text)]), ("learner", learner), ("reference", text), ("algorithm", algorithm), ("trajectories", P.MapType trajectory), ("behavior_model", record [("base", text), ("assembly", text)]), ("behavior", P.MapType (P.SequenceType P.BitsType)), ("reference_scores", P.MapType (P.SequenceType P.BitsType)), ("rewards", P.MapType P.NumberType), ("groups", selectors), ("order", selectors)]
  where
    selectors = P.SequenceType (P.MapType P.BooleanType)
    text = P.SequenceType P.TokenType
    record = P.RecordType . Map.fromList
    optimizer = record [("learning_rate", P.NumberType), ("betas", P.SequenceType P.NumberType), ("epsilon", P.NumberType), ("weight_decay", P.NumberType)]
    learner = record [("policy", text), ("learner", text), ("tokenizer", text), ("base", text), ("assembly", text), ("optimizer", optimizer)]
    algorithm = record [("epsilon", P.NumberType), ("penalty", P.NumberType), ("delta", P.NumberType), ("steps", P.NumberType)]
    trajectory = record [("prompt", text), ("seed", P.NumberType), ("limit", P.TokenType), ("temperature", P.NumberType), ("tokens", P.SequenceType P.TokenType), ("prompt_length", P.TokenType), ("text", text), ("truncated", P.BooleanType)]
