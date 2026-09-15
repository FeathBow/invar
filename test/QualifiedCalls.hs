{-# LANGUAGE OverloadedStrings #-}

module QualifiedCalls (qualifiedCalls) where

import BatchCalls qualified as Batch
import Calls qualified as F
import Control.Monad (forM_, void)
import Data.Aeson (Value (..), encode, object, toJSON, (.=))
import Data.ByteString.Lazy qualified as Lazy
import Data.Foldable (toList)
import Data.Maybe (listToMaybe)
import Hedgehog
import Invar.Infer qualified as Infer
import Invar.Infer.Invocation qualified as Call
import Invar.Learn.Protocol qualified as Learn
import Invar.Qualification qualified as Gate
import Invar.Spec.Evidence qualified as E
import Invar.Spec.Invocation qualified as I
import Invar.Spec.Qualification qualified as Q
import Updates qualified as U

qualifiedCalls :: Group
qualifiedCalls =
    Group
        "Required numerical qualification at actual dispatch"
        [ ("inference permit retains conditional evidence for the exact call", once inference)
        , ("learning completion retains its admitted certificate", once learning)
        , ("unknown region or composition cannot grant inference permission", once missing)
        , ("qualification materialization must match the prepared request", once materialization)
        , ("required profile inventory and methods reject incomplete declarations", once malformed)
        , ("batch certificates bind each distinct actual invocation", once batch)
        ]
  where
    once = withTests 1 . property

document :: [Value] -> Value
document profiles = object ["format" .= String "invar-numerical-qualification/v1", "profiles" .= profiles]

profile :: String -> Value -> Value
profile role actual =
    object
        [ "role" .= role
        , "tokenizer" .= F.field "tokenizer" actual
        , "base" .= F.field "base" actual
        , "assembly" .= F.field "assembly" actual
        , "target" .= defined (if role == "inference" then "categorical-inference-refinement/v1" else "grpo-update-refinement/v1")
        , "regions" .= map (\name -> F.change "method" (String "external-assumption") (defined (name ++ "-correspondence/v1"))) regions
        , "composition" .= String "external-assumption"
        ]
  where
    regions = ["materialization", "model-forward", "observation"] ++ if role == "inference" then ["tokenization", "sampling"] else ["scalar-objective", "model-differentiation", "optimizer-restoration"]
    defined predicate =
        object
            [ "predicate" .= (predicate :: String)
            , "reference" .= object ["name" .= String "protocol-control-fixture", "artifact" .= replicate 64 'a', "entry" .= String "fixture"]
            , "observation" .= String "Protocol admission only; no numerical correctness assertion"
            ]

registry :: Gate.Role -> Value -> PropertyT IO Gate.Registry
registry role = evalEither . Gate.decode role . Lazy.toStrict . encode

inference :: PropertyT IO ()
inference = do
    (call, events) <- F.setup
    firstEvent <- evalMaybe (listToMaybe events)
    selected <- registry Gate.Inference (document [profile "inference" firstEvent])
    (owner, permit) <- evalEither (Call.authorize selected call (F.wire (F.reviewPrefix events)))
    accepted <- evalMaybe (Call.qualified permit)
    Q.invocation (Q.qualifiedSubject accepted) === Call.binding call
    Gate.qualified (Gate.close owner) (Call.binding call) === Just accepted
    let proof = Q.certificate accepted
    length (E.assumptions proof) === 6
    assert (E.ImplicationElimination `elem` E.methods proof)
    assert (E.conclusion proof `notElem` E.assumptions proof)
    F.field "judgement" (Gate.report accepted) === String "conditional"
    F.field "assumptions" (Gate.report accepted) /== toJSON ([] :: [Value])
    (completed, _) <- evalEither (Call.observe permit (F.wire events))
    I.completedBinding completed === Q.invocation (Q.qualifiedSubject accepted)

learning :: PropertyT IO ()
learning = do
    (context, events) <- U.setup
    firstEvent <- evalMaybe (listToMaybe events)
    selected <- registry Gate.Learning (document [profile "learning" (F.field "state" firstEvent)])
    (_, permit) <- evalEither (Learn.authorize selected context (F.wire (take 2 events)))
    accepted <- evalMaybe (Learn.qualifiedPermit permit)
    length (E.assumptions (Q.certificate accepted)) === 7
    completed <- evalEither (Learn.observe permit (F.wire events))
    Learn.qualified completed === Just accepted
    Q.invocation (Q.qualifiedSubject accepted) === I.completedBinding (Learn.completion completed)

missing :: PropertyT IO ()
missing = do
    (call, events) <- F.setup
    firstEvent <- evalMaybe (listToMaybe events)
    let selected = profile "inference" firstEvent
        region = replaceFirstRegion (F.change "method" (String "unknown")) selected
        composition = F.change "composition" (String "unknown") selected
    forM_ [region, composition] $ \incomplete -> do
        owner <- registry Gate.Inference (document [incomplete])
        case Call.authorize owner call (F.wire (F.reviewPrefix events)) of
            Left (Call.Qualification (Gate.QualificationError (Q.Inconclusive (E.MissingEvidence _)))) -> success
            _ -> failure

materialization :: PropertyT IO ()
materialization = do
    (call, events) <- F.setup
    firstEvent <- evalMaybe (listToMaybe events)
    forM_ ["tokenizer", "base", "assembly"] $ \name -> do
        let selected = F.change name (toJSON (replicate 64 '0')) (profile "inference" firstEvent)
        owner <- registry Gate.Inference (document [selected])
        case Call.authorize owner call (F.wire (F.reviewPrefix events)) of
            Left (Call.Qualification (Gate.ProfileError _)) -> success
            _ -> failure

malformed :: PropertyT IO ()
malformed = do
    (_, events) <- F.setup
    firstEvent <- evalMaybe (listToMaybe events)
    let selected = profile "inference" firstEvent
        changed =
            [ F.change "regions" (toJSON ([] :: [Value])) selected
            , F.change "composition" (String "proved") selected
            , replaceFirstRegion (F.change "method" (String "test-passed")) selected
            ]
        documents = map (document . pure) changed ++ [document [selected, selected], document []]
    forM_ documents $ \supplied -> case Gate.decode Gate.Inference (Lazy.toStrict (encode supplied)) of
        Left (Gate.ProfileError _) -> success
        _ -> failure

batch :: PropertyT IO ()
batch = do
    (_, events) <- F.setup
    firstEvent <- evalMaybe (listToMaybe events)
    planned <- evalEither (Infer.prepare F.request)
    requests <- traverse (\index -> Batch.prepared planned events index Nothing) [0 .. 2]
    owner <- registry Gate.Inference (document [profile "inference" firstEvent])
    (updated, permits) <- evalEither (Call.authorizeBatch owner [(call, F.wire (F.reviewPrefix reported)) | (call, reported) <- requests])
    accepted <- traverse (evalMaybe . Call.qualified) permits
    map (Q.invocation . Q.qualifiedSubject) accepted === map (Call.binding . fst) requests
    forM_ (zip requests permits) $ \((call, reported), permit) -> do
        Gate.qualified updated (Call.binding call) === Call.qualified permit
        void (evalEither (Call.observe permit (F.wire reported)))

replaceFirstRegion :: (Value -> Value) -> Value -> Value
replaceFirstRegion change selected = case F.field "regions" selected of
    Array regions -> case toList regions of
        first : remaining -> F.change "regions" (toJSON (change first : remaining)) selected
        [] -> error "Qualification fixture needs a region"
    _ -> error "Qualification fixture regions must be an array"
