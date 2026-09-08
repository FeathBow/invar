{-# LANGUAGE Safe #-}

module Invar.Spec.Load (
    Registry,
    Descriptor (..),
    Image (..),
    Dispatch (..),
    Error (..),
    Fact,
    description,
    report,
    Live,
    empty,
    active,
    close,
    imageValue,
    expectedEmission,
    register,
    historical,
    acquire,
    unload,
    dispatch,
) where

import Control.Monad (unless, when)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Invocation qualified as I
import Invar.Spec.Value (Scalar (..), Value (..))
import Numeric.Natural (Natural)

data Image = Image {artifact :: ByteString, profile :: ByteString}
    deriving (Eq, Show)

data Descriptor = Descriptor {instanceName :: I.Instance, image :: Image}
    deriving (Eq, Show)

data Fact = Fact Descriptor I.Completion
    deriving (Eq, Show)

newtype Live = Live Fact
    deriving (Eq, Show)

data Dispatch = Dispatch {authorization :: Live, attempt :: I.Binding}

data Entry = Entry Fact Bool
newtype Registry = Registry (Map I.Instance Entry)

data Error
    = DuplicateInstance I.Instance
    | UnknownInstance I.Instance
    | Unloaded I.Instance
    | ReportInstanceMismatch I.Instance I.Instance
    | ReportEmissionMismatch E.Emission E.Emission
    | ReportArtifactMismatch ByteString ByteString
    | AuthorizationMismatch
    | DispatchInstanceMismatch I.Instance I.Instance
    | DispatchImageMismatch I.CallId (Value Natural) (Maybe (Value Natural))
    | InvocationError I.Error
    deriving (Eq, Show)

empty :: Registry
empty = Registry Map.empty

active :: Registry -> [I.Instance]
active (Registry entries) = Map.keys (Map.filter live entries)
  where
    live (Entry _ present) = present

close :: Registry -> Registry
close (Registry entries) = Registry (Map.map retired entries)
  where
    retired (Entry fact _) = Entry fact False

description :: Fact -> Descriptor
description (Fact descriptor _) = descriptor

report :: Fact -> I.Completion
report (Fact _ completed) = completed

expectedEmission :: Image -> E.Emission
expectedEmission = E.Emission "load" "policy-load/v1" . imageValue

imageValue :: Image -> Value Natural
imageValue loaded = Record (Map.fromList [("artifact", octets (artifact loaded)), ("profile", octets (profile loaded))])
  where
    octets = Sequence . map (Atom . Token . fromIntegral) . Bytes.unpack

register :: Descriptor -> I.Completion -> Registry -> Either Error Registry
register descriptor completed (Registry entries) = do
    let name = instanceName descriptor
        actual = I.boundInstance (I.completedBinding completed)
        expected = expectedEmission (image descriptor)
        received = I.completedEmission completed
        identity = artifact (image descriptor)
    when (Map.member name entries) (Left (DuplicateInstance name))
    unless (actual == name) (Left (ReportInstanceMismatch name actual))
    unless (received == expected) (Left (ReportEmissionMismatch expected received))
    unless (I.completedOutput completed == identity) (Left (ReportArtifactMismatch identity (I.completedOutput completed)))
    pure (Registry (Map.insert name (Entry (Fact descriptor completed) True) entries))

historical :: Registry -> I.Instance -> Either Error Fact
historical registry name = do
    Entry fact _ <- lookupEntry registry name
    pure fact

acquire :: Registry -> I.Instance -> Either Error Live
acquire registry name = do
    Entry fact present <- lookupEntry registry name
    unless present (Left (Unloaded name))
    pure (Live fact)

unload :: I.Instance -> Registry -> Either Error Registry
unload name registry@(Registry entries) = do
    Live fact <- acquire registry name
    pure (Registry (Map.insert name (Entry fact False) entries))

dispatch :: Dispatch -> Registry -> I.Runtime -> Either Error I.Runtime
dispatch request registry runtime = do
    let Live fact = authorization request
        name = instanceName (description fact)
        selected = I.boundInstance (attempt request)
    current <- acquire registry name
    unless (current == authorization request) (Left AuthorizationMismatch)
    unless (selected == name) (Left (DispatchInstanceMismatch name selected))
    intended <- either (Left . InvocationError) Right (I.intent runtime (I.boundCall (attempt request)))
    let expected = imageValue (image (description fact))
        actual = case E.payload intended of
            Record fields -> Map.lookup "policy" fields
            _ -> Nothing
    unless (actual == Just expected) (Left (DispatchImageMismatch (I.boundCall (attempt request)) expected actual))
    either (Left . InvocationError) Right (I.issue (attempt request) runtime)

lookupEntry :: Registry -> I.Instance -> Either Error Entry
lookupEntry (Registry entries) name = maybe (Left (UnknownInstance name)) Right (Map.lookup name entries)
