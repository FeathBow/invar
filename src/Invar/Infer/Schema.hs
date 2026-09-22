{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

module Invar.Infer.Schema (Inputs, inputs, policy) where

import Data.Map.Strict qualified as Map
import Invar.Construct qualified as C
import Invar.Spec.Program qualified as P
import Numeric.Natural (Natural)

type Inputs = C.Record '[ '("artifact", [Natural]), '("tokenizer", [Natural]), '("base", [Natural]), '("assembly", [Natural]), '("prompt", [Natural]), '("tokens", Natural), '("temperature", Rational)]

inputs :: P.Type
inputs = P.RecordType (Map.fromList [("artifact", text), ("tokenizer", text), ("base", text), ("assembly", text), ("prompt", text), ("tokens", P.TokenType), ("temperature", P.NumberType)])
  where
    text = P.SequenceType P.TokenType

policy :: P.Type
policy = P.RecordType (Map.fromList [("artifact", P.SequenceType P.TokenType), ("profile", P.SequenceType P.TokenType)])
