{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

module Invar.Load (Plan, Error (..), prepare, program, register, invocation) where

import Control.Monad (unless)
import Data.Aeson (Object, withObject, (.:))
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text.Encoding (encodeUtf8)
import Invar.Construct qualified as C
import Invar.Infer.Wire qualified as Wire
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as L
import Invar.Spec.Program qualified as P
import Numeric.Natural (Natural)

data Plan = Plan L.Descriptor V.Binding ByteString V.Runtime

data Error = Construction C.BuildError | Lifecycle V.Error | Registry L.Error | Protocol String
    deriving (Eq, Show)

prepare :: V.Binding -> L.Image -> Either Error Plan
prepare bound image = do
    checked <- either (Left . Construction) Right meaning
    let inputs = Map.singleton (P.Semantic "policy") (L.imageValue image)
    ready <- lifecycle (V.prepare (V.Selection (V.boundCall bound) inputs) (V.start checked 0))
    issued <- lifecycle (V.issue bound ready)
    pure (Plan (L.Descriptor (V.boundInstance bound) image) bound (A.bytes checked) issued)

meaning :: Either C.BuildError A.Checked
meaning = C.compile semantics [C.emit @"load" @"policy-load/v1" @'[ 'C.Semantic "policy"] expression]
  where
    expression = C.source @('C.Semantic "policy") @(C.Record '[ '("artifact", [Natural]), '("profile", [Natural])])
    policy = P.RecordType (Map.fromList [("artifact", P.SequenceType P.TokenType), ("profile", P.SequenceType P.TokenType)])
    sources = Map.singleton (P.Semantic "policy") policy
    sink = P.Sink "policy-load/v1" policy (Map.keysSet sources) Set.empty
    semantics = E.Semantics (P.Schema sources Map.empty (Map.singleton "load" sink)) Map.empty

program :: Plan -> ByteString
program (Plan _ _ encoded _) = encoded

register :: Plan -> Object -> L.Registry -> Either Error L.Registry
register planned@(Plan descriptor _ _ _) value registry = do
    completed <- completion planned value
    either (Left . Registry) Right (L.register descriptor completed registry)

completion :: Plan -> Object -> Either Error V.Completion
completion (Plan _ expected _ issued) value = do
    (bound, programBytes) <- parse (\fields -> fields .: "load" >>= withObject "policy load invocation" invocation) value
    actual <- parse (\fields -> fields .: "image" >>= withObject "loaded policy image" image) value
    unless (bound == expected) (Left (Lifecycle (V.BindingMismatch expected bound)))
    consumed <- lifecycle (V.consume (V.Consumption bound programBytes (L.expectedEmission actual)) issued)
    finished <- lifecycle (V.finish bound (L.artifact actual) consumed)
    reported <- lifecycle (V.completion finished (V.boundAttempt bound))
    maybe (Left (Protocol "Policy load did not complete")) Right reported
  where
    image fields = L.Image . encodeUtf8 <$> fields .: "artifact" <*> (encodeUtf8 <$> fields .: "profile")

invocation :: Object -> Parser (V.Binding, ByteString)
invocation fields = (,) <$> Wire.binding fields <*> (encodeUtf8 <$> fields .: "program")

parse :: (Object -> Parser value) -> Object -> Either Error value
parse parser = either (Left . Protocol) Right . parseEither parser

lifecycle :: Either V.Error value -> Either Error value
lifecycle = either (Left . Lifecycle) Right
