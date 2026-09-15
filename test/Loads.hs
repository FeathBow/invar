{-# LANGUAGE OverloadedStrings #-}

module Loads (loads, descriptor, bound, registered, callFor, runtimeFor) where

import Control.Monad (forM_)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Hedgehog
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Invocation qualified as I
import Invar.Spec.Load qualified as L
import Invar.Spec.Program
import Invar.Spec.Value (Scalar (..), Value (..))

loads :: Group
loads =
    Group
        "Load facts and live dispatch"
        [ ("registration retains the complete load report", once registration)
        , ("load reports bind instance artifact and profile", once reportBinding)
        , ("live dispatch preserves invocation checks", once liveDispatch)
        , ("dispatch matches the policy in the semantic emission", once dispatchTarget)
        , ("unload revokes handles but retains historical facts", once revocation)
        , ("reload requires a fresh instance identity", once reload)
        , ("same instance label does not replace a load fact", once foreignHandle)
        , ("unknown instances never produce live authorization", once missing)
        , ("process close retires every load and preserves historical facts", once closed)
        ]
  where
    once = withTests 1 . property

descriptor :: L.Descriptor
descriptor = L.Descriptor (I.Instance 0) (L.Image "policy-0" "numeric-profile-0")

bound :: I.Binding
bound = I.Binding (I.CallId 0) (I.AttemptId 0) (I.Instance 0)

emission :: L.Image -> E.Emission
emission loaded = E.Emission "load" "policy-load/v1" payload
  where
    payload = Record (Map.fromList [("artifact", bytes (L.artifact loaded)), ("profile", bytes (L.profile loaded))])
    bytes = Sequence . map (Atom . Token . fromIntegral) . Bytes.unpack

runtimeFor :: E.Emission -> Type -> PropertyT IO (A.Checked, I.Runtime)
runtimeFor selected kind = do
    let schema = Schema Map.empty Map.empty (Map.singleton (E.sinkName selected) (Sink (E.sinkSpecification selected) kind Set.empty Set.empty))
        command = Emit (E.sinkName selected) (E.sinkSpecification selected) (Constant kind (E.payload selected))
    checked <- evalEither (A.load (A.encode (E.Semantics schema Map.empty) [command]))
    ready <- evalEither (I.prepare (I.Selection (I.CallId 0) Map.empty) (I.start checked 0))
    pure (checked, ready)

completed :: L.Descriptor -> E.Emission -> ByteString -> PropertyT IO I.Completion
completed loaded selected output = do
    let kind = RecordType (Map.fromList [("artifact", SequenceType TokenType), ("profile", SequenceType TokenType)])
        binding = bound {I.boundInstance = L.instanceName loaded}
    (checked, ready) <- runtimeFor selected kind
    sent <- evalEither (I.issue binding ready)
    used <- evalEither (I.consume (I.Consumption binding (A.bytes checked) selected) sent)
    done <- evalEither (I.finish binding output used)
    evalEither (I.completion done (I.AttemptId 0)) >>= evalMaybe

registered :: L.Descriptor -> PropertyT IO L.Registry
registered loaded = do
    result <- completed loaded (emission (L.image loaded)) (L.artifact (L.image loaded))
    evalEither (L.register loaded result L.empty)

readyCall :: PropertyT IO I.Runtime
readyCall = callFor (L.image descriptor)

callFor :: L.Image -> PropertyT IO I.Runtime
callFor target = do
    let policy = E.payload (emission target)
        policyType = RecordType (Map.fromList [("artifact", SequenceType TokenType), ("profile", SequenceType TokenType)])
        payload = Record (Map.fromList [("policy", policy), ("value", Atom (Number 3))])
        kind = RecordType (Map.fromList [("policy", policyType), ("value", NumberType)])
    snd <$> runtimeFor (E.Emission "decode" "reference" payload) kind

reject :: L.Error -> Either L.Error value -> PropertyT IO ()
reject expected result = case result of
    Left actual -> actual === expected
    Right _ -> failure

registration :: PropertyT IO ()
registration = do
    result <- completed descriptor (emission (L.image descriptor)) "policy-0"
    registry <- evalEither (L.register descriptor result L.empty)
    fact <- evalEither (L.historical registry (I.Instance 0))
    L.description fact === descriptor
    L.report fact === result
    L.expectedEmission (L.image descriptor) === emission (L.image descriptor)

reportBinding :: PropertyT IO ()
reportBinding = do
    let expected = emission (L.image descriptor)
    wrongInstance <- completed descriptor {L.instanceName = I.Instance 1} expected "policy-0"
    reject (L.ReportInstanceMismatch (I.Instance 0) (I.Instance 1)) (L.register descriptor wrongInstance L.empty)
    forM_ [emission (L.Image "policy-1" "numeric-profile-0"), emission (L.Image "policy-0" "numeric-profile-1"), expected {E.sinkName = "other"}, expected {E.sinkSpecification = "other"}] $ \wrong -> do
        result <- completed descriptor wrong "policy-0"
        reject (L.ReportEmissionMismatch expected wrong) (L.register descriptor result L.empty)
    wrongOutput <- completed descriptor expected "policy-1"
    reject (L.ReportArtifactMismatch "policy-0" "policy-1") (L.register descriptor wrongOutput L.empty)

liveDispatch :: PropertyT IO ()
liveDispatch = do
    registry <- registered descriptor
    live <- evalEither (L.acquire registry (I.Instance 0))
    ready <- readyCall
    sent <- evalEither (L.dispatch (L.Dispatch live bound) registry ready)
    I.phase sent (I.AttemptId 0) === Right I.Issued
    reject (L.InvocationError (I.CallInUse (I.CallId 0) (I.AttemptId 0))) (L.dispatch (L.Dispatch live bound) registry sent)
    reject (L.DispatchInstanceMismatch (I.Instance 0) (I.Instance 1)) (L.dispatch (L.Dispatch live bound {I.boundInstance = I.Instance 1}) registry ready)

dispatchTarget :: PropertyT IO ()
dispatchTarget = do
    registry <- registered descriptor
    live <- evalEither (L.acquire registry (I.Instance 0))
    let expected = E.payload (emission (L.image descriptor))
    forM_ [L.Image "policy-1" "numeric-profile-0", L.Image "policy-0" "numeric-profile-1"] $ \wrong -> do
        ready <- callFor wrong
        reject (L.DispatchImageMismatch (I.CallId 0) expected (Just (E.payload (emission wrong)))) (L.dispatch (L.Dispatch live bound) registry ready)
    missingPolicy <- snd <$> runtimeFor (E.Emission "decode" "reference" (Atom (Number 3))) NumberType
    reject (L.DispatchImageMismatch (I.CallId 0) expected Nothing) (L.dispatch (L.Dispatch live bound) registry missingPolicy)

revocation :: PropertyT IO ()
revocation = do
    registry <- registered descriptor
    live <- evalEither (L.acquire registry (I.Instance 0))
    before <- evalEither (L.historical registry (I.Instance 0))
    unloaded <- evalEither (L.unload (I.Instance 0) registry)
    reject (L.Unloaded (I.Instance 0)) (L.acquire unloaded (I.Instance 0))
    ready <- readyCall
    reject (L.Unloaded (I.Instance 0)) (L.dispatch (L.Dispatch live bound) unloaded ready)
    reject (L.Unloaded (I.Instance 0)) (L.unload (I.Instance 0) unloaded)
    L.historical unloaded (I.Instance 0) === Right before

reload :: PropertyT IO ()
reload = do
    registry <- registered descriptor
    original <- evalEither (L.historical registry (I.Instance 0))
    reject (L.DuplicateInstance (I.Instance 0)) (L.register descriptor (L.report original) registry)
    unloaded <- evalEither (L.unload (I.Instance 0) registry)
    reject (L.DuplicateInstance (I.Instance 0)) (L.register descriptor (L.report original) unloaded)
    let replacement = descriptor {L.instanceName = I.Instance 1}
    result <- completed replacement (emission (L.image replacement)) "policy-0"
    reloaded <- evalEither (L.register replacement result unloaded)
    live <- evalEither (L.acquire reloaded (I.Instance 1))
    ready <- readyCall
    _ <- evalEither (L.dispatch (L.Dispatch live bound {I.boundInstance = I.Instance 1}) reloaded ready)
    L.historical reloaded (I.Instance 0) === Right original
    reject (L.Unloaded (I.Instance 0)) (L.acquire reloaded (I.Instance 0))

foreignHandle :: PropertyT IO ()
foreignHandle = do
    first <- registered descriptor
    second <- registered descriptor {L.image = L.Image "policy-1" "numeric-profile-0"}
    live <- evalEither (L.acquire first (I.Instance 0))
    ready <- readyCall
    reject L.AuthorizationMismatch (L.dispatch (L.Dispatch live bound) second ready)

missing :: PropertyT IO ()
missing = do
    reject (L.UnknownInstance (I.Instance 0)) (L.acquire L.empty (I.Instance 0))
    reject (L.UnknownInstance (I.Instance 0)) (L.unload (I.Instance 0) L.empty)
    reject (L.UnknownInstance (I.Instance 0)) (L.historical L.empty (I.Instance 0))

closed :: PropertyT IO ()
closed = do
    registry <- registered descriptor
    first <- evalEither (L.historical registry (I.Instance 0))
    let replacement = descriptor {L.instanceName = I.Instance 1}
    second <- completed replacement (emission (L.image replacement)) "policy-0"
    both <- evalEither (L.register replacement second registry)
    L.active both === [I.Instance 0, I.Instance 1]
    let retired = L.close both
    L.active retired === []
    L.active (L.close retired) === []
    L.historical retired (I.Instance 0) === Right first
    forM_ [I.Instance 0, I.Instance 1] $ \name -> reject (L.Unloaded name) (L.acquire retired name)
    reject (L.DuplicateInstance (I.Instance 1)) (L.register replacement second retired)
