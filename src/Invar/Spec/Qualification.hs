{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE Safe #-}

module Invar.Spec.Qualification (
    Subject,
    Definition (..),
    Premises (..),
    Policy,
    Key (..),
    Registry,
    Request,
    QualifiedResult,
    Error (..),
    subject,
    actualLoad,
    invocation,
    operation,
    bindings,
    policy,
    empty,
    register,
    revoke,
    close,
    request,
    conclusion,
    requiredRegions,
    regionClaims,
    composition,
    permittedAssumptions,
    qualify,
    certificate,
    qualifiedSubject,
    dispatch,
) where

import Control.Monad (unless, when)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.List (find)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text qualified as Text
import Data.Text.Encoding (encodeUtf8)
import Invar.Spec.Evidence qualified as E
import Invar.Spec.Invocation qualified as I
import Invar.Spec.Load qualified as L
import Numeric.Natural (Natural)

data Subject = Subject L.Fact I.Binding I.Intention
    deriving (Eq, Show)

data Definition = Definition {predicate :: String, reference :: ByteString, observation :: String}
    deriving (Eq, Show)

data Premises = Premises {regionAssumptions :: [Definition], compositionAssumed :: Bool}
    deriving (Eq, Show)

data Policy = Policy Definition [Definition] Premises
    deriving (Eq, Show)

newtype Key = Key Natural
    deriving (Eq, Ord, Show)

data Entry = Entry Policy Bool
newtype Registry = Registry (Map Key Entry)

data Request = Request Key Policy Subject E.Claim [E.Claim] E.Claim [E.Claim]
    deriving (Eq, Show)

data QualifiedResult = QualifiedResult Request E.Certificate
    deriving (Eq, Show)

data Error
    = LoadError L.Error
    | InvocationError I.Error
    | MalformedPolicy String
    | DuplicateKey Key
    | UnknownKey Key
    | Revoked Key
    | ChangedPolicy Key
    | Inconclusive E.Problem
    | Refuted E.Counterexample
    | WrongConclusion E.Claim E.Claim
    | UnapprovedHypothesis E.Claim
    | SubjectMismatch
    deriving (Eq, Show)

subject :: L.Registry -> I.Binding -> I.Runtime -> Either Error Subject
subject registry bound runtime = fst <$> inspect registry bound runtime

inspect :: L.Registry -> I.Binding -> I.Runtime -> Either Error (Subject, I.Runtime)
inspect registry bound runtime = do
    live <- first LoadError (L.acquire registry (I.boundInstance bound))
    issued <- first LoadError (L.dispatch (L.Dispatch live bound) registry runtime)
    selected <- first InvocationError (I.intention runtime (I.boundCall bound))
    loaded <- first LoadError (L.historical registry (I.boundInstance bound))
    pure (Subject loaded bound selected, issued)

actualLoad :: Subject -> L.Fact
actualLoad (Subject loaded _ _) = loaded

invocation :: Subject -> I.Binding
invocation (Subject _ bound _) = bound

operation :: Subject -> I.Intention
operation (Subject _ _ selected) = selected

policy :: Definition -> [Definition] -> Premises -> Either Error Policy
policy target required allowed = do
    let assumed = regionAssumptions allowed
    when (null required) (Left (MalformedPolicy "Qualification requires a nonempty region inventory"))
    mapM_ validate (target : required)
    unless (distinct (map predicate required)) (Left (MalformedPolicy "Qualification predicates must be distinct"))
    when (predicate target `elem` map predicate required) (Left (MalformedPolicy "The whole-operation predicate must differ from its primitive regions"))
    unless (distinct (map predicate assumed) && all (`elem` required) assumed) (Left (MalformedPolicy "Declared assumptions must be distinct exact required regions"))
    pure (Policy target required allowed)
  where
    distinct values = length values == Set.size (Set.fromList values)
    validate defined = unless (not (null (predicate defined)) && not (Bytes.null (reference defined)) && not (null (observation defined))) (Left (MalformedPolicy "A region must name its predicate reference and observation"))

empty :: Registry
empty = Registry Map.empty

register :: Key -> Policy -> Registry -> Either Error Registry
register key selected (Registry entries) = do
    when (Map.member key entries) (Left (DuplicateKey key))
    pure (Registry (Map.insert key (Entry selected True) entries))

revoke :: Key -> Registry -> Either Error Registry
revoke key registry@(Registry entries) = do
    selected <- active registry key
    pure (Registry (Map.insert key (Entry selected False) entries))

close :: Registry -> Registry
close (Registry entries) = Registry (Map.map (\(Entry selected _) -> Entry selected False) entries)

active :: Registry -> Key -> Either Error Policy
active (Registry entries) key = case Map.lookup key entries of
    Nothing -> Left (UnknownKey key)
    Just (Entry _ False) -> Left (Revoked key)
    Just (Entry selected True) -> Right selected

request :: Registry -> Key -> Subject -> Either Error Request
request registry key selected = do
    declared@(Policy target required allowed) <- active registry key
    let loaded = actualLoad selected
        instantiate = claim (bindings selected)
        observedLoad = E.OutputEqual (L.report loaded) (L.artifact (L.image (L.description loaded)))
        regions = observedLoad : map instantiate required
        expected = instantiate target
        implication = E.Implies (E.All regions) expected
        assumed = [implication | compositionAssumed allowed] ++ map instantiate (regionAssumptions allowed)
    pure (Request key declared selected expected regions implication assumed)

claim :: (ByteString, ByteString) -> Definition -> E.Claim
claim (domain, binding) defined = E.External (E.Obligation (predicate defined) (reference defined) (observation defined) domain binding)

bindings :: Subject -> (ByteString, ByteString)
bindings selected = (domain, binding)
  where
    domain = "invar-qualification-input/v1\NUL" <> encoded (operation selected)
    binding = "invar-qualification-binding/v1\NUL" <> encoded (actualLoad selected, invocation selected)
    encoded value = encodeUtf8 (Text.pack (show value))

conclusion :: Request -> E.Claim
conclusion (Request _ _ _ expected _ _ _) = expected

requiredRegions :: Request -> E.Claim
requiredRegions = E.All . regionClaims

regionClaims :: Request -> [E.Claim]
regionClaims (Request _ _ _ _ required _ _) = required

composition :: Request -> E.Claim
composition (Request _ _ _ _ _ implication _) = implication

permittedAssumptions :: Request -> [E.Claim]
permittedAssumptions (Request _ _ _ _ _ _ allowed) = allowed

qualify :: Request -> E.Graph -> E.EvidenceId -> Either Error QualifiedResult
qualify requested graph root = case E.check graph root of
    E.Unknown problem -> Left (Inconclusive problem)
    E.Refute counterexample ->
        if E.refuted counterexample == conclusion requested
            then Left (Refuted counterexample)
            else Left (WrongConclusion (conclusion requested) (E.refuted counterexample))
    E.Accept accepted -> do
        let expected = conclusion requested
            actual = E.conclusion accepted
        unless (actual == expected) (Left (WrongConclusion expected actual))
        case find (`notElem` permittedAssumptions requested) (E.assumptions accepted) of
            Just unexpected -> Left (UnapprovedHypothesis unexpected)
            Nothing -> pure (QualifiedResult requested accepted)

certificate :: QualifiedResult -> E.Certificate
certificate (QualifiedResult _ accepted) = accepted

qualifiedSubject :: QualifiedResult -> Subject
qualifiedSubject (QualifiedResult (Request _ _ selected _ _ _ _) _) = selected

dispatch :: (Registry, L.Registry) -> QualifiedResult -> I.Runtime -> Either Error I.Runtime
dispatch (authority, loads) (QualifiedResult (Request key declared original _ _ _ _) _) runtime = do
    current <- active authority key
    unless (current == declared) (Left (ChangedPolicy key))
    (selected, issued) <- inspect loads (invocation original) runtime
    unless (selected == original) (Left SubjectMismatch)
    pure issued
