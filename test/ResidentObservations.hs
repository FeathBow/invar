{-# LANGUAGE OverloadedStrings #-}

module ResidentObservations (residentObservations, fixture, reseal) where

import Calls (change, field, request, wire)
import Control.Monad (foldM, forM_)
import Data.Aeson (Value (..), eitherDecodeStrict, object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8, encodeUtf8)
import Hedgehog
import Invar.Artifact qualified as Artifact
import Invar.Evaluation qualified as Evaluation
import Invar.Infer qualified as Infer
import Invar.Workload qualified as Workload
import Store (workspace)
import Streams qualified
import System.FilePath ((</>))
import Updates (alter)
import Workloads (array, encoded)

residentObservations :: Group
residentObservations =
    Group
        "Complete resident evaluation observations"
        [ ("one physical load spans all cohorts including unused owners", once complete)
        , ("each original group release and physical close is mandatory", once boundaries)
        , ("physical declarations owners and clocks remain exact", once ownership)
        , ("a matching wire digest cannot replace semantic program admission", once semantic)
        ]
  where
    once = withTests 1 . property

cohortCount, membersPerCohort :: Int
cohortCount = 3
membersPerCohort = 2

run :: Evaluation.Run
run = Evaluation.Run (Infer.artifact request) 0

fixture :: Int -> PropertyT IO (Workload.Document, [Value])
fixture owners = do
    (original, events, _) <- Streams.batched
    tasks <- evalEither (Workload.decode (encoded (toJSON (concat (replicate cohortCount (array (Workload.value original)))))))
    case events of
        [loading, profile, loaded, ready, timed, finished, summary, completed] -> do
            let reported = change "model" (String "fixture") (change "revision" (String "fixture") profile)
                cohort index = concatMap (group index) [0 .. owners - 1] ++ [summaryFor index summary]
                group index slot =
                    let selected = [member | member <- [0 .. membersPerCohort - 1], member `mod` owners == slot]
                        select frame = change "calls" (toJSON [rebind (index * membersPerCohort + member) value | (member, value) <- zip [0 ..] (array (field "calls" frame)), member `elem` selected]) frame
                        prefix = if index == 0 then [loading, reported, loaded] else [timer "activation" activationSeconds]
                     in if null selected then [] else prefix ++ [select ready, timed, select finished, acknowledgement slot]
                tailRecords = [closed slot (if slot < membersPerCohort then cohortCount else 0) | slot <- reverse [0 .. owners - 1]]
                completion = change "worker_mode" (String "resident") (change "sessions" (toJSON owners) (change "cohorts" (toJSON cohortCount) (change "tasks_sha256" (toJSON (Workload.digest tasks)) completed)))
            sealed <- reseal (concatMap cohort [0 .. cohortCount - 1] ++ tailRecords ++ [completion])
            pure (tasks, sealed)
        _ -> failure

activationSeconds, releaseSeconds, closeSeconds :: Double
activationSeconds = 0.5
releaseSeconds = 0.25
closeSeconds = 0.125

timer :: Text -> Double -> Value
timer operation seconds = object ["stage" .= operation, "cpu_seconds" .= seconds]

owner :: Int -> Value
owner slot = object ["role" .= String "inference", "session" .= slot]

acknowledgement :: Int -> Value
acknowledgement slot = object ["stage" .= String "released", "format" .= String "invar-resident-v1", "owner" .= owner slot, "measurement" .= decodeUtf8 (wire [timer "released" releaseSeconds])]

closed :: Int -> Int -> Value
closed slot count = object ["stage" .= String "closed", "format" .= String "invar-resident-v1", "owner" .= owner slot, "groups" .= count, "measurement" .= decodeUtf8 (wire [timer "closed" closeSeconds])]

bound :: Int -> Value
bound index = object ["call" .= index, "attempt" .= index, "instance" .= index]

rebind :: Int -> Value -> Value
rebind index (String source) = String (decodeUtf8 (wire (map replace (decode (encodeUtf8 source)))))
  where
    replace (Object fields) = Object (Fields.insert "binding" (bound index) (Fields.mapWithKey nested fields))
    replace _ = error "Expected resident fixture call object"
    nested "load" value = change "binding" (bound index) value
    nested _ value = value
rebind _ _ = error "Expected original batch member JSONL"

decode :: ByteString -> [Value]
decode = map (either error id . eitherDecodeStrict) . Bytes.lines

summaryFor :: Int -> Value -> Value
summaryFor index value = change "cohort" (toJSON index) (change "samples" (toJSON samples) value)
  where
    samples = zipWith (\member -> change "binding" (bound (index * membersPerCohort + member))) [0 ..] (array (field "samples" value))

reseal :: [Value] -> PropertyT IO [Value]
reseal events = do
    root <- workspace
    (_, accepted) <- foldM (step (root </> "group.jsonl")) ([], []) events
    pure (reverse accepted)
  where
    step path (pending, accepted) value
        | stage "released" value = do
            let preceding = reverse pending
                consumed = [call | row <- preceding, stage "consumed" row, member <- array (field "calls" row), call <- memberRows member, stage "consumed" call]
            evalIO (Bytes.writeFile path (wire preceding))
            digest <- evalIO (Artifact.identity "Offline resident group fixture" path)
            let receipt = change "result_sha256" (toJSON digest) (change "loads" (toJSON (map (field "load") consumed)) value)
            pure ([], receipt : accepted)
        | phase value || stage "closed" value = pure ([], value : accepted)
        | otherwise = pure (value : pending, value : accepted)
    memberRows (String raw) = decode (encodeUtf8 raw)
    memberRows _ = error "Expected consumed resident member fixture"

stage :: Text -> Value -> Bool
stage expected (Object fields) = Fields.lookup "stage" fields == Just (String expected)
stage _ _ = False

phase :: Value -> Bool
phase (Object fields) = Fields.member "phase" fields
phase _ = False

complete :: PropertyT IO ()
complete = forM_ [1, 2, 3] $ \owners -> do
    (tasks, events) <- fixture owners
    report <- evalEither (Evaluation.admit tasks run (wire events))
    length (Evaluation.samples report) === cohortCount * membersPerCohort
    Evaluation.inputDigest report === Workload.digest tasks
    assert (case Evaluation.residence report of Just _ -> True; Nothing -> False)

boundaries :: PropertyT IO ()
boundaries = do
    (tasks, events) <- fixture 2
    forM_ ["load", "activation", "consumed", "inference", "result", "released", "closed"] $ \name -> do
        index <- evalMaybe (firstIndex (stage name) events)
        reject tasks (take index events ++ drop (index + 1) events)
        reject tasks (take index events ++ [events !! index] ++ drop index events)
    reject tasks (events ++ [timer "activation" activationSeconds])
    rejected (Evaluation.admit tasks run (Bytes.init (wire events)))

ownership :: PropertyT IO ()
ownership = do
    (tasks, events) <- fixture 3
    let mutate name operation = do index <- evalMaybe (firstIndex (stage name) events); reject tasks (alter index operation events)
    mutate "released" (change "owner" (owner 9))
    mutate "released" (change "loads" (toJSON ([] :: [Value])))
    mutate "released" (change "measurement" (String "{}\n"))
    mutate "closed" (change "groups" (Number 1))
    mutate "closed" (change "owner" (owner 0))
    mutate "activation" (change "cpu_seconds" (Number (-1)))
    let lastIndex = length events - 1
    reject tasks (alter lastIndex (change "sessions" (Number 2)) events)
    reject tasks (alter lastIndex (change "worker_mode" (String "serial")) events)
    rejected (Evaluation.admit tasks (run {Evaluation.exitCode = 7}) (wire events))

semantic :: PropertyT IO ()
semantic = do
    (tasks, events) <- fixture 1
    index <- evalMaybe (firstIndex (stage "consumed") events)
    let altered frame = change "calls" (toJSON (alter 0 member (array (field "calls" frame)))) frame
        member (String raw) = String (decodeUtf8 (wire (map (\row -> if stage "consumed" row then change "program" (String "different semantic program") row else row) (decode (encodeUtf8 raw)))))
        member _ = error "Expected consumed batch fixture"
    changed <- reseal (alter index altered events)
    reject tasks changed

firstIndex :: (value -> Bool) -> [value] -> Maybe Int
firstIndex predicate values = case [index | (index, value) <- zip [0 ..] values, predicate value] of
    index : _ -> Just index
    [] -> Nothing

reject :: Workload.Document -> [Value] -> PropertyT IO ()
reject tasks = rejected . Evaluation.admit tasks run . wire

rejected :: Either String value -> PropertyT IO ()
rejected (Left _) = success
rejected (Right _) = failure
