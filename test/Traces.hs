{-# LANGUAGE OverloadedStrings #-}

module Traces (traces, built, admit) where

import BatchCalls (quote)
import Calls qualified
import Control.Monad (forM_)
import Data.Aeson (Value (..), eitherDecodeStrict, encode, object, toJSON, withObject, (.:), (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Either (fromRight, isLeft)
import Data.IORef (atomicModifyIORef', newIORef, readIORef)
import Data.List (isInfixOf)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8)
import Hedgehog
import Invar.Artifact qualified as Artifact
import Invar.Async.Entry qualified as Entry
import Invar.Cohort qualified as C
import Invar.History.Cohort qualified as Cohort
import Invar.History.Generation qualified as Generation
import Invar.History.Trace qualified as Trace
import Invar.Infer qualified as I
import Invar.Journal qualified as Journal
import Invar.Learn qualified as L
import Invar.Learn.Report qualified as Report
import Invar.Learn.Worker qualified as Learner
import Invar.Loop qualified as Loop
import Invar.Policy qualified as Policy
import Invar.Rollout qualified as R
import Invar.Runtime qualified as Runtime
import Invar.Spec.Invocation qualified as V
import Invar.Store qualified as Store
import Invar.Transcript qualified as Transcript
import Invar.Worker qualified as Worker
import Invar.Workload qualified as Workload
import LearnerFixture qualified as F
import Numeric.Natural (Natural)
import ResidentWorkloads qualified as Rollout
import Sessions qualified
import Store (workspace)
import System.Directory (createDirectory)
import System.FilePath ((</>))

traces :: Group
traces =
    Group
        "Training history replay"
        [ ("a history replays each finite session, the learner and its publication under the declared policy", once admitted)
        , ("a history whose learner produces the reference words, instead of taking the engine's scores, is admitted with that source and refused without those words", once scoredByLearner)
        , ("a shared history carries the inference group and the learner in one owner's records and closes that owner once", once shared)
        , ("a generation is replayed under its declared policy description and reference, never under the worker's report", once declared)
        , ("a resident learner release is admitted by its acknowledgement's owner, loads and result digest", once released)
        , ("a session cut before its final result, or output after the history, is refused", once incomplete)
        , ("a two-cycle training log admits each generation from its predecessor's publication and ends at the second generation's policy", once twoCycles)
        ]
  where
    once = withTests 1 . property

data History = History {run :: Trace.Run, initial :: Policy.Description, document :: Workload.Document, inference :: [ByteString], learning :: [Value], closing :: [Value], published :: Value, finished :: Value}

assembled :: History -> ByteString
assembled history = Bytes.unlines (inference history ++ map line (learning history ++ [published history, finished history] ++ closing history))

line :: Value -> ByteString
line = Lazy.toStrict . encode

decoded :: ByteString -> Value
decoded = fromRight Null . eitherDecodeStrict

has :: Key -> Value -> Bool
has key (Object fields) = Fields.member key fields
has _ _ = False

admit :: History -> Either String Trace.Checked
admit history = Trace.admit (run history) (initial history) (document history) (assembled history)

describedBy :: L.Settings -> PropertyT IO Policy.Description
describedBy settings = evalEither (Policy.describe ("test-model", "test-revision") (L.policy settings, L.tokenizer settings, L.behaviorBase settings, L.behaviorAssembly settings))

workload :: R.Options -> PropertyT IO Workload.Document
workload chosen = evalEither (Workload.decode (line (toJSON [object ["tasks" .= map task (C.tasks (R.definition chosen)), "order" .= R.order chosen, "delivery" .= R.delivery chosen]])))
  where
    task member = object ["name" .= C.name member, "group" .= String "group", "prompt" .= I.prompt Calls.request, "tokens" .= I.tokens Calls.request, "temperature" .= I.temperature Calls.request, "seed" .= I.seed Calls.request, "answer" .= String "#### 12"]

publication :: FilePath -> (Natural, Natural) -> [Natural] -> Value -> Value
publication output (index, update) delivery result = object ["phase" .= String "published", "checkpoint" .= (output </> ("generation" ++ show (index + 1))), "policy" .= Calls.field "adapter" result, "learner" .= Calls.field "learner" result, "publication" .= String "rename", "binding" .= ordinal update, "delivery" .= map ordinal delivery]
  where
    ordinal number = object ["call" .= number, "attempt" .= number, "instance" .= number]

summary :: Natural -> Value
summary index = object ["phase" .= String "cycle", "index" .= index, "sessions" .= (1 :: Int), "seconds" .= (0.5 :: Double)]

built :: Int -> Bool -> PropertyT IO History
built count = builtFrom F.configured (`Sessions.options` count) count

builtFrom :: L.Settings -> (FilePath -> PropertyT IO R.Options) -> Int -> Bool -> PropertyT IO History
builtFrom settings selected count resident = do
    root <- workspace
    chosen <- selected root
    (description, sessions, (before, steps, after, acknowledged)) <- recorded root settings R.Serial chosen
    result <- final after
    let output = root </> "train"
        prefix = if resident then before else filter ((/= Just "activation") . F.stage) before
        closed = object ["stage" .= String "closed", "format" .= String "invar-resident-v1", "owner" .= F.owner, "groups" .= (1 :: Int), "measurement" .= decodeUtf8 (F.wire [F.timer "closed"])]
    declaration <- workload chosen
    pure
        History
            { run = Trace.Run settings (fromIntegral count) output "rename" 0 Trace.Finite (if resident then Trace.Resident else Trace.Finite)
            , initial = description
            , document = declaration
            , inference = sessions
            , learning = prefix ++ steps ++ after ++ [acknowledged | resident]
            , closing = [closed | resident]
            , published = publication output (0, fromIntegral (length (C.tasks (R.definition chosen)))) (R.delivery chosen) result
            , finished = object ["phase" .= String "cycle", "index" .= (0 :: Int), "sessions" .= count, "seconds" .= (0.5 :: Double)]
            }

recorded :: FilePath -> L.Settings -> R.Mode -> R.Options -> PropertyT IO (Policy.Description, [ByteString], ([Value], [Value], [Value], Value))
recorded root settings mode chosen = do
    description <- describedBy settings
    let definition = R.definition chosen
    bound <- evalEither (traverse (\task -> either (Left . show) (\planned -> Right task {C.plan = planned}) (I.bindPolicy description (C.plan task))) (C.tasks definition))
    captured <- evalIO (newIORef Map.empty)
    let opened slot = pure (Transcript.Transcript (\written -> atomicModifyIORef' captured (\held -> (Map.insertWith (flip (++)) slot [written] held, ()))) (const (pure ())) (const (pure ())))
        update = V.ordinal (fromIntegral (length bound))
    produced <- evalIO $ R.withRecordedDriver mode (R.worker chosen, R.sessions chosen) opened $ \driver -> do
        batch <- R.run driver chosen {R.definition = definition {C.tasks = bound}} >>= F.require
        planned <- F.require (L.prepare settings batch)
        exchange <- F.prepare root (0, update) planned
        pure (F.before exchange, F.steps exchange, F.after exchange, F.released exchange)
    learned <- evalEither produced
    sessions <- evalIO (readIORef captured)
    pure (description, concat (Map.elems sessions), learned)

final :: [Value] -> PropertyT IO Value
final after = case reverse after of
    value : _ -> pure value
    [] -> failure

admitted :: PropertyT IO ()
admitted = forM_ [(1, False), (2, False), (1, True)] $ \(count, resident) -> do
    history <- built count resident
    checked <- evalEither (admit history)
    case (Trace.generations checked, Workload.cycles (document history)) of
        ([generation], [declaration]) -> length (Cohort.inferences (Generation.cohort generation)) === length (Workload.tasks declaration)
        _ -> failure
    toJSON (Policy.adapter (Trace.finalPolicy checked)) === Calls.field "policy" (published history)

scoredByLearner :: PropertyT IO ()
scoredByLearner = forM_ [False, True] $ \resident -> do
    history <- builtFrom settings scored 1 resident
    checked <- evalEither (admit history)
    sources <- evalEither (traverse (parseEither (withObject "update request" (.: "reference_source")) . Report.request . Cohort.update . Generation.cohort) (Trace.generations checked))
    sources === ["learner" :: Text]
    assert (isLeft (admit history {learning = [record | record <- learning history, F.stage record /= Just "reference"]}))
  where
    settings = F.configured {L.referenceSource = L.FromLearner, L.reference = replicate 64 'c'}
    scored root = do
        chosen <- Sessions.optionsWith root 1 0 (L.policy settings, Just (L.reference settings))
        pure chosen {R.reference = Loop.scoring (configuredBy root chosen settings root) (Loop.Checkpoint "" (L.policy settings) (L.learner settings))}

shared :: PropertyT IO ()
shared = do
    history <- sharedHistory
    checked <- evalEither (admit history)
    length (Trace.generations checked) === 1
    assert (isLeft (admit history {closing = []}))
    case admit history {run = (run history) {Trace.learningMode = Trace.Resident}} of
        Left problem -> problem === "Shared history requires both roles and exactly one physical session"
        Right _ -> failure

sharedHistory :: PropertyT IO History
sharedHistory = do
    root <- workspace
    chosen <- Rollout.initial <$> Rollout.setup root 1 [L.policy settings]
    (description, hosting, (before, steps, after, acknowledged)) <- recorded root settings R.Resident chosen
    result <- final after
    let owner = object ["role" .= String "shared", "session" .= (0 :: Int)]
        relabeled encoded = if has "owner" (decoded encoded) then line (Calls.change "owner" owner (decoded encoded)) else encoded
        hosted = [relabeled encoded | encoded <- hosting, F.stage (decoded encoded) /= Just "closed"]
        aligned record = if F.stage record == Just "loaded_learner" then foldr (uncurry Calls.change) record [("model", String "test-model"), ("revision", String "test-revision")] else record
        group = [aligned record | record <- before ++ steps ++ after, F.stage record /= Just "load"]
        output = root </> "train"
    evalIO (Bytes.writeFile (root </> "learning.jsonl") (F.wire group))
    digest <- evalIO (Artifact.identity "Shared learner records" (root </> "learning.jsonl"))
    declaration <- workload chosen
    pure
        History
            { run = Trace.Run settings 1 output "rename" 0 Trace.Shared Trace.Shared
            , initial = description
            , document = declaration
            , inference = hosted
            , learning = group ++ [Calls.change "owner" owner (Calls.change "result_sha256" (String (Text.pack digest)) acknowledged)]
            , closing = [object ["stage" .= String "closed", "format" .= String "invar-resident-v1", "owner" .= owner, "groups" .= (2 :: Int), "measurement" .= decodeUtf8 (F.wire [F.timer "closed"])]]
            , published = publication output (0, fromIntegral (length (C.tasks (R.definition chosen)))) (R.delivery chosen) result
            , finished = summary 0
            }
  where
    settings = F.configured

declared :: PropertyT IO ()
declared = do
    history <- built 1 False
    let settings = Trace.settings (run history)
    other <- evalEither (Policy.describe ("other-model", "test-revision") (Policy.bindings (initial history)))
    assert (isLeft (admit history {initial = other}))
    case admit history {run = (run history) {Trace.settings = settings {L.reference = replicate 64 'd'}}} of
        Left problem -> assert ("Reference scores differ from the declared reference" `isInfixOf` problem)
        Right _ -> failure

released :: PropertyT IO ()
released = do
    history <- built 1 True
    _ <- evalEither (admit history)
    let altered = [if F.stage record == Just "released" then Calls.change "result_sha256" (String (Text.replicate 64 "0")) record else record | record <- learning history]
    assert (altered /= learning history)
    assert (isLeft (admit history {learning = altered}))

incomplete :: PropertyT IO ()
incomplete = do
    history <- built 2 False
    _ <- evalEither (admit history)
    assert (isLeft (admit history {inference = take (length (inference history) - 1) (inference history)}))
    assert (isLeft (admit history {closing = closing history ++ [object ["stage" .= String "result"]]}))

twoCycles :: PropertyT IO ()
twoCycles = do
    checked <- twoCycleLog
    adapters <- evalEither (traverse (Report.artifact "adapter" . Cohort.update . Generation.cohort) (Trace.generations checked))
    case adapters of
        [earlier, later] -> do
            earlier /== L.policy F.configured
            earlier /== later
            Policy.adapter (Trace.finalPolicy checked) === later
        _ -> failure

configuredBy :: FilePath -> R.Options -> L.Settings -> FilePath -> Loop.Config
configuredBy base opening settings root = Loop.Config backend root (base </> "input") (base </> "reference") settings Store.RenameExclusive
  where
    worker = R.worker opening
    backend = Loop.Backend "/bin/sh" (Worker.executable worker) (base </> "inference.sh") Nothing R.Serial (base </> "learner.sh") Nothing Learner.Process (Worker.cache worker) (R.sessions opening) Learner.Process

twoCycleLog :: PropertyT IO Trace.Checked
twoCycleLog = do
    base <- workspace
    let settings = F.configured
    opening <- Sessions.options base 1
    let members = fromIntegral (length (C.tasks (R.definition opening)))
    description <- describedBy settings
    evalIO $ do
        createDirectory (base </> "input")
        Policy.stageDescription (base </> "input" </> "policy.json") description
    firstLearner <- learned opening (base, 0, V.ordinal members) (settings, description, Nothing)
    adapter <- evalIO (Policy.identity (base </> "update0" </> "adapter.safetensors"))
    learner <- evalIO (Artifact.identity "Learner checkpoint" (base </> "update0" </> "learner.pt"))
    successor <- evalEither (Policy.successor adapter description)
    following <- Sessions.optionsWith base 1 (members + 1) (adapter, Just (L.reference settings))
    evalIO (createDirectory (base </> "prepared"))
    preparing <- Sessions.optionsWith (base </> "prepared") 1 0 (adapter, Just (L.reference settings))
    let worker = R.worker opening
        inferring = base </> "inference.sh"
        output = base </> "runtime"
        config = configuredBy base opening settings
        updated = settings {L.policy = adapter, L.learner = learner, L.schedule = L.synchronous 1 adapter}
        scoring = Loop.scoring (config base) (Loop.Checkpoint "" adapter learner)
    secondLearner <- learned preparing (base, 1, V.ordinal (2 * members + 1)) (updated, successor, scoring)
    let copying index = unlines ["output=\"${4#--output=}\"", "mkdir -p \"$output\" || exit 24", unwords (["cp"] ++ [quote (base </> ("update" ++ show (index :: Int)) </> name) | name <- ["adapter.safetensors", "learner.pt", "gradients.safetensors", "probabilities.json"]] ++ ["\"$output\"", "|| exit 25"])]
        dispatching marker (firstScript, laterScript) = unlines ["if test -e " ++ quote marker ++ "; then exec /bin/sh " ++ quote laterScript ++ " \"$@\"; fi", ": > " ++ quote marker, "exec /bin/sh " ++ quote firstScript ++ " \"$@\""]
    evalIO $ do
        writeFile (base </> "learner0.sh") (copying 0 ++ firstLearner)
        writeFile (base </> "learner1.sh") (copying 1 ++ secondLearner)
        writeFile (base </> "learner.sh") (dispatching (base </> "learned") (base </> "learner0.sh", base </> "learner1.sh"))
        writeFile inferring (dispatching (base </> "inferred") (Worker.script worker, Worker.script (R.worker following)))
    declaration <- evalEither (Sessions.workloads 2)
    outcome <- evalIO (Runtime.run (Runtime.Run (config output) 0 declaration) Null)
    either (\problem -> annotateShow problem >> failure) pure outcome
    journaled <- evalIO (Bytes.readFile (output </> "journal.jsonl"))
    entries <- evalEither (Journal.entries journaled >>= traverse (parseEither Entry.decode))
    let reserved = [(number, role) | Entry.Reserved number role _ _ <- entries]
    map snd reserved === [Entry.Inference, Entry.Learner, Entry.Inference, Entry.Learner]
    transcripts <- evalIO (traverse (\(number, _) -> Bytes.lines <$> Bytes.readFile (output </> "transcripts" </> (show number ++ ".jsonl"))) reserved)
    let offset index = index * (members + 1)
        result learning = [decoded encoded | encoded <- learning, F.stage (decoded encoded) == Just "result", has "gradients" (decoded encoded)]
        cycled (index, (sessions, learning)) = sessions ++ learning ++ map line ([publication output (index, offset index + members) (map (offset index +) (R.delivery opening)) reported | reported <- result learning] ++ [summary index])
    evalEither (Trace.admit (Trace.Run settings 1 output "rename" 0 Trace.Finite Trace.Finite) description declaration (Bytes.unlines (concatMap cycled (zip [0 ..] (pairs transcripts)))))
  where
    pairs (inferred : learnt : rest) = (inferred, learnt) : pairs rest
    pairs _ = []
    learned options (root, index, binding) (chosen, policy, reference) = do
        bound <- evalEither (first show (Loop.bindTasks policy (C.tasks (R.definition options))))
        prepared <- evalIO $ R.withConfiguredDriver R.Serial (R.worker options, R.sessions options) $ \driver -> do
            batch <- R.run driver options {R.definition = C.Definition (L.policy chosen) bound, R.reference = reference} >>= F.require
            planned <- F.require (L.prepare chosen batch)
            F.process (root </> "replies") <$> F.prepare root (index, binding) planned
        evalEither prepared
