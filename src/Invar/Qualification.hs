module Invar.Qualification (
    Role (..),
    Registry,
    Error (..),
    empty,
    open,
    decode,
    loads,
    withLoads,
    close,
    dispatch,
    qualified,
    report,
    emit,
) where

import Data.Aeson (Value)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Invar.Qualification.Profile (Role (..))
import Invar.Qualification.Profile qualified as Profile
import Invar.Qualification.Report (emit)
import Invar.Qualification.Report qualified as Report
import Invar.Spec.Invocation qualified as I
import Invar.Spec.Load qualified as L
import Invar.Spec.Qualification qualified as Q

data Registry = Registry L.Registry (Maybe Profile.Profile) Q.Registry (Map I.AttemptId Q.QualifiedResult)

data Error = RegistryError L.Error | QualificationError Q.Error | ProfileError String
    deriving (Eq, Show)

profileKey :: Q.Key
profileKey = Q.Key 0

empty :: Registry
empty = Registry L.empty Nothing Q.empty Map.empty

open :: Role -> Maybe FilePath -> IO Registry
open _ Nothing = pure empty
open selected (Just path) = do
    encoded <- Bytes.readFile path
    either (ioError . userError . show) pure (decode selected encoded)

decode :: Role -> ByteString -> Either Error Registry
decode selected encoded = do
    profile <- first ProfileError (Profile.decode selected encoded)
    authority <- first QualificationError (Q.register profileKey (Profile.policy profile) Q.empty)
    pure (Registry L.empty (Just profile) authority Map.empty)

loads :: Registry -> L.Registry
loads (Registry registry _ _ _) = registry

withLoads :: L.Registry -> Registry -> Registry
withLoads updated (Registry _ profile authority retained) = Registry updated profile authority retained

close :: Registry -> Registry
close (Registry registry profile authority retained) = Registry (L.close registry) profile (Q.close authority) retained

dispatch :: Registry -> I.Binding -> I.Runtime -> Either Error (Registry, I.Runtime)
dispatch registry@(Registry current profile authority retained) bound runtime = case profile of
    Nothing -> do
        live <- first RegistryError (L.acquire current (I.boundInstance bound))
        issued <- first RegistryError (L.dispatch (L.Dispatch live bound) current runtime)
        pure (registry, issued)
    Just selected -> do
        subject <- first QualificationError (Q.subject current bound runtime)
        first ProfileError (Profile.verify selected (Q.operation subject))
        requested <- first QualificationError (Q.request authority profileKey subject)
        let (graph, root) = Profile.proof selected requested
        accepted <- first QualificationError (Q.qualify requested graph root)
        issued <- first QualificationError (Q.dispatch (authority, current) accepted runtime)
        pure (Registry current profile authority (Map.insert (I.boundAttempt bound) accepted retained), issued)

qualified :: Registry -> I.Binding -> Maybe Q.QualifiedResult
qualified (Registry _ _ _ retained) bound = do
    result <- Map.lookup (I.boundAttempt bound) retained
    if Q.invocation (Q.qualifiedSubject result) == bound then Just result else Nothing

report :: Q.QualifiedResult -> Value
report = Report.value
