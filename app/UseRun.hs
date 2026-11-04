{-# LANGUAGE OverloadedStrings #-}

module UseRun (run) where

import Check qualified
import Control.Exception (SomeException, displayException, try)
import Control.Monad (unless)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Result (..), Value (..), encode, fromJSON, object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Bifunctor (first)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Foldable (toList)
import Data.List (genericLength, mapAccumL)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe, mapMaybe)
import Data.Text qualified as Text
import Data.Text.Encoding qualified as TextBytes
import Envelope (Problem (..), command, refuse, succeed)
import Envelope qualified
import GHC.Clock (getMonotonicTime)
import InferenceInput qualified
import Invar.Artifact qualified as Artifact
import Invar.Infer.Observation qualified as Inference
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Session qualified as Session
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Numerical qualified as N
import Invar.Policy qualified as Policy
import Invar.Score qualified as Score
import Invar.Score.Worker qualified as ScoreWorker
import Invar.Spec.Invocation qualified as V
import Invar.Transcript qualified as Transcript
import Invar.Use qualified as U
import Invar.Use.Execution qualified as E
import Invar.Use.Prepare qualified as Prepare
import Invar.Worker qualified as Worker
import Numeric.Natural (Natural)
import Options qualified as O
import System.Console.GetOpt (OptDescr)
import System.Directory (createDirectory, doesDirectoryExist, doesFileExist, getFileSize, listDirectory, makeAbsolute)
import System.Environment (getExecutablePath)
import System.Exit (ExitCode (..))
import System.FilePath (takeDirectory, (</>))
import System.IO (IOMode (..), withBinaryFile)
import System.Process (StdStream (UseHandle), createProcess, proc, std_err, std_in, std_out, waitForProcess)
import UseExecution qualified as Execution
import UseInput (readContract, readDeclared)

format :: String
format = "invar-use-run-v1"

data Group = Group {label :: String, sideName :: String, schedule :: String, arrangement :: E.Arrangement, side :: Execution.Side, policy :: Policy.Description, members :: [(U.Input, Natural)]}

run :: [String] -> IO ()
run = command format "Usage: invar use run --contract FILE --execution FILE --output DIRECTORY\nRun a contract's finite execution plan on this host and keep every record. Prints one JSON document; read status and problems, then run invar use admit inside the output directory." options execution

execution :: [String] -> IO ()
execution supplied = do
    fields <- either (\found -> refuse format [Problem "missing-argument" "argv" found]) pure (O.parse options supplied)
    contractPath <- required fields "contract"
    executionPath <- required fields "execution"
    output <- required fields "output"
    Envelope.output format "a run directory" output
    (contract, contractBytes) <- readContract format "contract" contractPath
    (executionValue, executionBytes) <- readDeclared format "execution" executionPath
    let report = Prepare.units contract
        invalid = [Problem "artifact-invalid" ("artifact:" ++ contractPath) ("The contract is not valid: " ++ reason) | U.InvalidContract reason <- U.validate contract]
        relied = map U.premise (U.reliance contract)
        missing = [Problem "missing-reliance" ("artifact:" ++ contractPath ++ ":reliance") ("No reliance is declared for " ++ show kind ++ ", which this contract and execution plan raise") | kind <- Prepare.expectedPremises contract, kind `notElem` relied]
        decoded = Execution.decode executionValue `Check.andThen` \plan -> Execution.schedules (Prepare.inputCount report) (Prepare.candidateExecutions report) plan `Check.andThen` \() -> Execution.collectionCheck contract plan `Check.andThen` \() -> Check.value plan
        refused = invalid ++ missing
    declared <- either (refuse format . (refused ++)) pure (Check.run decoded)
    unless (null refused) (refuse format refused)
    plan <- resolved (takeDirectory executionPath) declared
    artifacts contract plan >>= \problems -> unless (null problems) (refuse format problems)
    let planned = groups contract plan
    createDirectory output
    createDirectory (output </> "logs")
    Bytes.writeFile (output </> "contract.json") contractBytes
    Lazy.writeFile (output </> "plan.json") (encode (object ["format" .= ("invar-use-run-plan-v1" :: String), "execution_sha256" .= digest executionBytes, "collection" .= describeCollection contract plan planned, "groups" .= map describe planned]))
    executable <- getExecutablePath
    started <- getMonotonicTime
    (records, reports) <- execute executable output plan planned
    collected <- collect output contract plan reports records >>= either (\problem -> refuse format [problem {message = message problem ++ "; every record so far is kept in " ++ (output </> "records.json")}]) pure
    finished <- getMonotonicTime
    Lazy.writeFile (output </> "runs.json") (encode (rows contract plan planned collected))
    cost <- costOf (realToFrac (finished - started)) output planned collected
    Lazy.writeFile (output </> "cost.json") (encode cost)
    succeed
        format
        "recorded"
        "Every record the contract needs under this execution plan was produced and retained."
        "That the candidate is admitted."
        [ "output" .= output
        , "contract_sha256" .= digest contractBytes
        , "execution_sha256" .= digest executionBytes
        , "groups" .= length records
        , "cost" .= cost
        , "files" .= (["contract.json", "plan.json", "records.json", "runs.json", "cost.json"] :: [String])
        , "next" .= ("cd " ++ output ++ " && invar use admit --contract contract.json --runs runs.json")
        ]
  where
    required fields name = maybe (refuse format [Problem "missing-argument" ("argv:--" ++ name) ("--" ++ name ++ " is required")]) pure (O.optional fields name)

resolved :: FilePath -> Execution.Execution -> IO Execution.Execution
resolved base plan = do
    python <- absolute (Execution.python plan)
    cache <- absolute (Execution.cache plan)
    reference <- side (Execution.reference plan)
    candidate <- side (Execution.candidate plan)
    scorers <- traverse scorer (E.scorers (Execution.collection plan))
    pure
        plan
            { Execution.python = python
            , Execution.cache = cache
            , Execution.reference = reference
            , Execution.candidate = candidate
            , Execution.collection = (Execution.collection plan) {E.scorers = scorers}
            }
  where
    absolute path = makeAbsolute (base </> path)
    side selected = Execution.Side <$> absolute (Execution.worker selected) <*> absolute (Execution.configuration selected) <*> absolute (Execution.adapter selected)
    scorer selected = do
        worker <- absolute (E.scorerWorker selected)
        configuration <- absolute (E.scorerConfiguration selected)
        adapter <- absolute (E.scorerAdapter selected)
        pure selected {E.scorerWorker = worker, E.scorerConfiguration = configuration, E.scorerAdapter = adapter}

artifacts :: U.UseContract -> Execution.Execution -> IO [Problem]
artifacts contract plan = do
    python <- file "execution:python" (Execution.python plan)
    cache <- directory "execution:cache" (Execution.cache plan)
    reference <- side "reference" (Execution.reference plan) (U.referenceImplementation contract)
    candidate <- side "candidate" (Execution.candidate plan) (U.candidateImplementation contract)
    scorers <- concat <$> traverse scorer (Map.toList (E.scorers (Execution.collection plan)))
    pure (python ++ cache ++ reference ++ candidate ++ scorers)
  where
    scorer (name, selected) = do
        let at field = "execution:collection.scorers." ++ pathName name ++ "." ++ field
            described = case name of N.Reference -> U.referenceImplementation contract; N.Candidate -> U.candidateImplementation contract
        worker <- file (at "worker") (E.scorerWorker selected)
        configuration <- file (at "configuration") (E.scorerConfiguration selected)
        adapter <- file (at "adapter") (E.scorerAdapter selected)
        identity <- first displayException <$> (try (Policy.identity (E.scorerAdapter selected)) :: IO (Either SomeException String))
        pure
            ( worker
                ++ configuration
                ++ adapter
                ++ case identity of
                    Left found -> [Problem "artifact-invalid" (at "adapter") ("The adapter could not be read: " ++ found)]
                    Right actual -> [Problem "artifact-invalid" (at "adapter") ("The adapter's identity " ++ actual ++ " differs from the " ++ pathName name ++ " policy's adapter " ++ Policy.adapter described) | actual /= Policy.adapter described]
            )
    file at path = (\found -> [Problem "artifact-missing" at (path ++ " does not exist") | not found]) <$> doesFileExist path
    directory at path = (\found -> [Problem "artifact-missing" at (path ++ " is not a directory") | not found]) <$> doesDirectoryExist path
    side name selected described = do
        let at field = "execution:sides." ++ name ++ "." ++ field
        worker <- file (at "worker") (Execution.worker selected)
        configuration <- file (at "configuration") (Execution.configuration selected)
        adapter <- file (at "adapter") (Execution.adapter selected)
        identity <-
            if null adapter
                then first displayException <$> (try (Policy.identity (Execution.adapter selected)) :: IO (Either SomeException String))
                else pure (Right (Policy.adapter described))
        pure
            ( worker
                ++ configuration
                ++ adapter
                ++ case identity of
                    Left found -> [Problem "artifact-invalid" (at "adapter") ("The adapter could not be read: " ++ found)]
                    Right actual -> [Problem "artifact-invalid" (at "adapter") ("The adapter's identity " ++ actual ++ " differs from the " ++ name ++ " policy's adapter " ++ Policy.adapter described) | actual /= Policy.adapter described]
            )

groups :: U.UseContract -> Execution.Execution -> [Group]
groups contract plan = snd (mapAccumL number 0 planned)
  where
    inputs = toList (U.declaredInputs (U.declaredDomain contract))
    runs =
        [ ("reference", "reference", "paired", Execution.paired plan, Execution.reference plan, U.referenceImplementation contract)
        , ("candidate", "candidate", "paired", Execution.paired plan, Execution.candidate plan, U.candidateImplementation contract)
        ]
            ++ [("repeat" ++ show index, "candidate", "repeats[" ++ show index ++ "]", arranged, Execution.candidate plan, U.candidateImplementation contract) | (index, arranged) <- zip [0 :: Int ..] (Execution.repeats plan)]
    planned = [(name ++ "-" ++ show ordinal, owner, scheduled, arranged, selected, described, batch) | (name, owner, scheduled, arranged, selected, described) <- runs, (ordinal, batch) <- zip [0 :: Int ..] (E.arrange arranged inputs)]
    number next (name, owner, scheduled, arranged, selected, described, batch) = (next + genericLength batch, Group name owner scheduled arranged selected described (zip batch [next ..]))

request :: Policy.Description -> U.Input -> Natural -> [(String, String)]
request described input call =
    [ ("digest", adapter)
    , ("tokenizer-digest", tokenizer)
    , ("base-digest", base)
    , ("assembly-digest", assembly)
    , ("prompt", U.prompt input)
    , ("tokens", show (U.tokens input))
    , ("temperature", show (U.temperature input))
    , ("seed", show (U.seed input))
    , ("call", show call)
    , ("attempt", show call)
    , ("instance", show call)
    ]
  where
    (adapter, tokenizer, base, assembly) = Policy.bindings described

flags :: [(String, String)] -> [String]
flags pairs = concat [["--" ++ name, entry] | (name, entry) <- pairs]

logPath :: Group -> FilePath
logPath group = "logs" </> (label group ++ ".jsonl")

callsPath :: Group -> FilePath
callsPath group = "logs" </> (label group ++ "-calls.json")

data Collected = Collected
    { collectedKind :: String
    , collectedInput :: U.Input
    , collectedPath :: N.Side
    , collectedTarget :: N.Side
    , collectedSteps :: [Natural]
    , collectedBinding :: V.Binding
    , collectedLog :: FilePath
    , collectedExit :: Int
    , collectedSourceLog :: String
    , collectedTargetPolicy :: Policy.Description
    , collectedLogDigest :: String
    , collectedSeconds :: Double
    , collectedPeakMemory :: Maybe Integer
    }

execute :: FilePath -> FilePath -> Execution.Execution -> [Group] -> IO ([Value], Map.Map (U.Key, N.Side) Inference.Report)
execute executable output plan = go [] Map.empty
  where
    go done reports [] = pure (done, reports)
    go done reports (group : rest) = do
        (outcome, record) <- once group
        let kept = done ++ [record]
            reports' = case outcome of
                Right admitted | schedule group == "paired" -> foldr (\((input, _), report) -> Map.insert (U.inputKey input, groupSide group) report) reports (zip (members group) admitted)
                _ -> reports
        Lazy.writeFile (output </> "records.json") (encode kept)
        either (\found -> refuse format [found {message = message found ++ "; every record so far is kept in " ++ (output </> "records.json")}]) (const (go kept reports' rest)) outcome
    once group = do
        let calls = callsPath group
            errors = "logs" </> (label group ++ ".stderr")
        Lazy.writeFile (output </> calls) (encode [flags (request (policy group) input call) | (input, call) <- members group])
        status <- launch group calls errors (["infer", "batch"] ++ worker group)
        logBytes <- readIfPresent (output </> logPath group)
        callBytes <- Bytes.readFile (output </> calls)
        let record =
                object
                    [ "group" .= label group
                    , "side" .= sideName group
                    , "schedule" .= schedule group
                    , "inputs" .= [(U.cohort (U.inputKey input), U.task (U.inputKey input)) | (input, _) <- members group]
                    , "calls" .= map snd (members group)
                    , "calls_file" .= calls
                    , "calls_sha256" .= digest callBytes
                    , "log" .= logPath group
                    , "log_sha256" .= digest logBytes
                    , "stderr" .= errors
                    , "exit_code" .= either (either (const Nothing) Just) Just status
                    ]
        pure (checked group (callBytes, logBytes) =<< first (failure group errors) status, record)
    worker group =
        flags
            [ ("python", Execution.python plan)
            , ("cache", Execution.cache plan)
            , ("worker", Execution.worker (side group))
            , ("adapter", Execution.adapter (side group))
            , ("worker-config", Execution.configuration (side group))
            ]
    launch group calls errors arguments = do
        started <-
            try
                ( withBinaryFile (output </> calls) ReadMode $ \input ->
                    withBinaryFile (output </> logPath group) WriteMode $ \stdout ->
                        withBinaryFile (output </> errors) WriteMode $ \stderr -> do
                            (_, _, _, handle) <- createProcess (proc executable arguments) {std_in = UseHandle input, std_out = UseHandle stdout, std_err = UseHandle stderr}
                            waitForProcess handle
                ) ::
                IO (Either SomeException ExitCode)
        pure $ case started of
            Left found -> Left (Left (displayException found))
            Right ExitSuccess -> Right 0
            Right (ExitFailure code) -> Left (Right code)
    failure group errors reason = Problem "execution-failed" (located group) $ case reason of
        Left found -> "Group " ++ label group ++ " could not start: " ++ found
        Right code -> "Group " ++ label group ++ " exited with " ++ show code ++ " on inputs " ++ keys group ++ "; see " ++ (output </> errors)
    readIfPresent path = doesFileExist path >>= \found -> if found then Bytes.readFile path else pure Bytes.empty
    groupSide group = case sideName group of
        "reference" -> N.Reference
        _ -> N.Candidate

checked :: Group -> (Bytes.ByteString, Bytes.ByteString) -> Int -> Either Problem [Inference.Report]
checked group (callBytes, logBytes) status = do
    declared <- first (Problem "internal-error" (located group) . ("The group's calls could not be read back: " ++)) (InferenceInput.calls callBytes)
    admitted <- first (Problem "execution-failed" (located group) . (("The log of group " ++ label group ++ " does not carry its declared batch: ") ++) . show) (Replay.standalone Session.Batched (Session.Declaration declared Nothing) (Replay.declared status) logBytes)
    traverse (member admitted) (members group)
  where
    member admitted (input, call) = do
        let binding = V.Binding (V.CallId call) (V.AttemptId call) (V.Instance call)
            named = label group ++ " input " ++ show (U.cohort (U.inputKey input), U.task (U.inputKey input))
        report <- case filter ((== binding) . Trajectory.binding . Replay.trajectory) admitted of
            [single] -> pure (Inference.view single)
            _ -> Left (Problem "execution-failed" (located group) ("The log does not carry the declared observation for " ++ named))
        reported <- first (Problem "identity-mismatch" (located group) . (("The worker's report for " ++ named ++ " names no policy: ") ++)) (Inference.policyDescription report)
        unless (reported == policy group) (Left (Problem "identity-mismatch" (located group) ("The worker reported " ++ show reported ++ " for " ++ named ++ ", not the contract's " ++ sideName group ++ " policy")))
        pure report

pathName :: N.Side -> String
pathName N.Reference = "reference"
pathName N.Candidate = "candidate"

data CollectionSpec = CollectionSpec
    { specKind :: String
    , specInput :: U.Input
    , specPath :: N.Side
    , specTarget :: N.Side
    , specSteps :: [Natural]
    , specCall :: Natural
    }

collectionSpecs :: U.UseContract -> Execution.Execution -> Natural -> [CollectionSpec]
collectionSpecs contract plan firstCall = zipWith assign [firstCall ..] specs
  where
    inputs = toList (U.declaredInputs (U.declaredDomain contract))
    specs =
        [CollectionSpec "score" input path (otherSide path) [] 0 | path <- E.scorePaths (Execution.collection plan), input <- inputs]
            ++ [CollectionSpec "probe" input (E.probePath probe) target (E.probeSteps probe) 0 | probe <- E.probes (Execution.collection plan), input <- inputs, target <- [N.Reference, N.Candidate]]
    assign call spec = spec {specCall = call}
    otherSide N.Reference = N.Candidate
    otherSide N.Candidate = N.Reference

collectionStart :: [Group] -> Natural
collectionStart planned = case concatMap members planned of
    [] -> 0
    values -> maximum (map snd values) + 1

collect :: FilePath -> U.UseContract -> Execution.Execution -> Map.Map (U.Key, N.Side) Inference.Report -> [Value] -> IO (Either Problem [Collected])
collect output contract plan reports records = loop [] records specs
  where
    planned = groups contract plan
    specs = collectionSpecs contract plan (collectionStart planned)
    loop :: [Collected] -> [Value] -> [CollectionSpec] -> IO (Either Problem [Collected])
    loop done _ [] = pure (Right (reverse done))
    loop done kept (spec : rest) = do
        (outcome, record) <- once spec
        let kept' = kept ++ [record]
        Lazy.writeFile (output </> "records.json") (encode kept')
        case outcome of
            Left problem -> pure (Left problem)
            Right collected -> loop (collected : done) kept' rest
    once spec = do
        case Map.lookup (U.inputKey (specInput spec), specPath spec) reports of
            Nothing -> pure (Left (Problem "execution-failed" "execution:collection" ("No admitted inference for " ++ show (U.inputKey (specInput spec)) ++ " and the " ++ pathName (specPath spec) ++ " path")), collectionRecord spec Nothing Nothing Nothing Nothing Nothing)
            Just admitted -> do
                let sourceLog = Inference.logDigest admitted
                    targetPolicy = case specTarget spec of N.Reference -> U.referenceImplementation contract; N.Candidate -> U.candidateImplementation contract
                    target = Score.Target (Policy.bindings targetPolicy) (Just targetPolicy)
                    scorer = Map.lookup (specTarget spec) (E.scorers (Execution.collection plan))
                case scorer of
                    Nothing -> pure (Left (Problem "execution-failed" "execution:collection" ("No scorer for the " ++ pathName (specTarget spec) ++ " implementation")), collectionRecord spec Nothing (Just sourceLog) (Just targetPolicy) Nothing Nothing)
                    Just selected -> case Score.prepare (Inference.source admitted) target of
                        Left problem -> pure (Left (Problem "execution-failed" "execution:collection" (show problem)), collectionRecord spec Nothing (Just sourceLog) (Just targetPolicy) Nothing Nothing)
                        Right basePlan -> case if specKind spec == "probe" then Score.withProbe (specSteps spec) basePlan else Right basePlan of
                            Left problem -> pure (Left (Problem "execution-failed" "execution:collection" (show problem)), collectionRecord spec Nothing (Just sourceLog) (Just targetPolicy) Nothing Nothing)
                            Right plannedScore -> case Score.bind (V.ordinal (specCall spec)) plannedScore of
                                Left problem -> pure (Left (Problem "execution-failed" "execution:collection" (show problem)), collectionRecord spec Nothing (Just sourceLog) (Just targetPolicy) Nothing Nothing)
                                Right call -> do
                                    let worker = Worker.Worker (Execution.python plan) (E.scorerWorker selected) (Execution.cache plan) (E.scorerAdapter selected) [] (Just (E.scorerConfiguration selected))
                                        scoreLog = collectionLog spec
                                    started <- getMonotonicTime
                                    returned <- withBinaryFile (output </> scoreLog) WriteMode $ \handle -> ScoreWorker.run worker (Transcript.file handle) call
                                    finished <- getMonotonicTime
                                    logDigest <- readLog (output </> scoreLog)
                                    let seconds = realToFrac (finished - started)
                                        logRecord = collectionRecord spec (either (Just . exitCode) (const (Just 0)) returned) (Just sourceLog) (Just targetPolicy) logDigest (Just seconds)
                                    case returned of
                                        Left failure -> pure (Left (Problem "execution-failed" (collectionAt spec) (show failure ++ "; see " ++ (output </> scoreLog))), logRecord)
                                        Right report ->
                                            let collected =
                                                    Collected
                                                        { collectedKind = specKind spec
                                                        , collectedInput = specInput spec
                                                        , collectedPath = specPath spec
                                                        , collectedTarget = specTarget spec
                                                        , collectedSteps = specSteps spec
                                                        , collectedBinding = V.ordinal (specCall spec)
                                                        , collectedLog = scoreLog
                                                        , collectedExit = 0
                                                        , collectedSourceLog = sourceLog
                                                        , collectedTargetPolicy = targetPolicy
                                                        , collectedLogDigest = fromMaybe "" logDigest
                                                        , collectedSeconds = seconds
                                                        , collectedPeakMemory = peakMemory report
                                                        }
                                             in pure (Right collected, logRecord)
    collectionLog spec = "logs/collection-" ++ show (specCall spec) ++ ".jsonl"
    collectionAt spec = "execution:collection." ++ specKind spec ++ "(" ++ show (U.inputKey (specInput spec)) ++ ")"
    readLog path = doesFileExist path >>= \found -> if found then Just . digest <$> Bytes.readFile path else pure Nothing
    exitCode (ScoreWorker.WorkerExit (ExitFailure code)) = code
    exitCode (ScoreWorker.WorkerExit ExitSuccess) = 0
    exitCode _ = 1

collectionRecord :: CollectionSpec -> Maybe Int -> Maybe String -> Maybe Policy.Description -> Maybe String -> Maybe Double -> Value
collectionRecord spec status sourceLog targetPolicy logDigest seconds =
    object
        [ "kind" .= specKind spec
        , "input" .= object ["cohort" .= U.cohort (U.inputKey (specInput spec)), "task" .= U.task (U.inputKey (specInput spec))]
        , "path" .= pathName (specPath spec)
        , "target" .= pathName (specTarget spec)
        , "steps" .= specSteps spec
        , "call" .= specCall spec
        , "log" .= ("logs/collection-" ++ show (specCall spec) ++ ".jsonl")
        , "source_log" .= sourceLog
        , "target_policy" .= fmap policyValue targetPolicy
        , "log_sha256" .= logDigest
        , "seconds" .= seconds
        , "exit_code" .= fmap show status
        ]

policyValue :: Policy.Description -> Value
policyValue described =
    object
        [ "model" .= Policy.model described
        , "revision" .= Policy.revision described
        , "adapter" .= Policy.adapter described
        , "tokenizer" .= Policy.tokenizer described
        , "base" .= Policy.base described
        , "assembly" .= Policy.assembly described
        ]

peakMemory :: Score.Report -> Maybe Integer
peakMemory report = case Score.describe report of
    Object top -> case Fields.lookup "observation" top of
        Just (Object observation) -> case Fields.lookup "measurements" observation of
            Just (Array measurements) -> maximumMaybe (mapMaybe measured (toList measurements))
            _ -> Nothing
        _ -> Nothing
    _ -> Nothing
  where
    measured (Object fields) = case Fields.lookup "peak_active" fields >>= number of
        Just value -> Just value
        Nothing -> case Fields.lookup "workers" fields of
            Just (Array workers) -> maximumMaybe (mapMaybe worker (toList workers))
            _ -> Nothing
    measured _ = Nothing
    worker (Object fields) = max <$> numberField "peak_allocated" fields <*> numberField "peak_reserved" fields
    worker _ = Nothing
    numberField name fields = Fields.lookup name fields >>= number
    number value = case (fromJSON value :: Result Integer) of
        Success found -> Just found
        _ -> Nothing
    maximumMaybe [] = Nothing
    maximumMaybe values = Just (maximum values)

costOf :: Double -> FilePath -> [Group] -> [Collected] -> IO Value
costOf wall output planned collected = do
    records <- Bytes.readFile (output </> "records.json")
    logs <- listDirectory (output </> "logs")
    logBytes <- sum <$> traverse (\name -> getFileSize (output </> "logs" </> name)) logs
    let peaks = mapMaybe collectedPeakMemory collected
        peak = case peaks of [] -> Null; values -> toJSON (maximum values)
        memoryStatus = if null peaks then ("unmeasured" :: String) else "score-probe-peak"
    pure
        ( object
            [ "inference_calls" .= sum [length (members group) | group <- planned]
            , "score_calls" .= length [value | value <- collected, collectedKind value == "score"]
            , "probe_calls" .= length [value | value <- collected, collectedKind value == "probe"]
            , "wall_seconds" .= wall
            , "records_bytes" .= Bytes.length records
            , "logs_bytes" .= logBytes
            , "peak_rss_bytes" .= peak
            , "memory_status" .= memoryStatus
            , "calls" .= [object ["kind" .= collectedKind value, "call" .= callOf (collectedBinding value), "seconds" .= collectedSeconds value] | value <- collected]
            ]
        )
  where
    callOf (V.Binding (V.CallId call) _ _) = call

describeCollection :: U.UseContract -> Execution.Execution -> [Group] -> Value
describeCollection contract plan planned =
    object
        [ "scores" .= map pathName (E.scorePaths collection)
        , "probes" .= [object ["path" .= pathName (E.probePath probe), "steps" .= E.probeSteps probe] | probe <- E.probes collection]
        , "calls" .= [object ["kind" .= specKind spec, "input" .= object ["cohort" .= U.cohort (U.inputKey (specInput spec)), "task" .= U.task (U.inputKey (specInput spec))], "path" .= pathName (specPath spec), "target" .= pathName (specTarget spec), "steps" .= specSteps spec, "call" .= specCall spec, "scorer" .= fmap scorerValue (Map.lookup (specTarget spec) (E.scorers collection))] | spec <- collectionSpecs contract plan (collectionStart planned)]
        ]
  where
    collection = Execution.collection plan
    scorerValue selected =
        object
            [ "backend" .= E.scorerBackend selected
            , "worker" .= E.scorerWorker selected
            , "configuration" .= E.scorerConfiguration selected
            , "adapter" .= E.scorerAdapter selected
            ]

located :: Group -> String
located group = "execution:" ++ schedule group

keys :: Group -> String
keys group = show [(U.cohort (U.inputKey input), U.task (U.inputKey input)) | (input, _) <- members group]

describe :: Group -> Value
describe group =
    object
        [ "group" .= label group
        , "side" .= sideName group
        , "schedule" .= schedule group
        , "order" .= case E.order (arrangement group) of
            E.Declared -> object ["order" .= ("declared" :: String)]
            E.Reversed -> object ["order" .= ("reversed" :: String)]
            E.Rotated offset -> object ["order" .= ("rotated" :: String), "offset" .= offset]
        , "group_size" .= E.groupSize (arrangement group)
        , "inputs" .= [(U.cohort (U.inputKey input), U.task (U.inputKey input)) | (input, _) <- members group]
        , "calls" .= map snd (members group)
        , "worker" .= Execution.worker (side group)
        , "configuration" .= Execution.configuration (side group)
        , "adapter" .= Execution.adapter (side group)
        , "policy" .= let (adapter, tokenizer, base, assembly) = Policy.bindings (policy group) in object ["adapter" .= adapter, "tokenizer" .= tokenizer, "base" .= base, "assembly" .= assembly]
        , "log" .= logPath group
        ]

rows :: U.UseContract -> Execution.Execution -> [Group] -> [Collected] -> [Value]
rows contract plan planned collected =
    [ encodeRow (U.inputKey input) (paired input) [executed ("repeats[" ++ show index ++ "]") input | index <- [0 .. length (Execution.repeats plan) - 1]]
    | input <- toList (U.declaredInputs (U.declaredDomain contract))
    ]
  where
    placed = Map.fromList [((schedule group, sideName group, U.inputKey input), (group, call)) | group <- planned, (input, call) <- members group]
    lookupRun scheduled owner input = placed Map.! (scheduled, owner, U.inputKey input)
    record (group, call) input = request (policy group) input call ++ [("log", logPath group), ("calls", callsPath group), ("exit-code", "0")]
    paired input = flags ([("reference-" ++ name, entry) | (name, entry) <- record (lookupRun "paired" "reference" input) input] ++ [("candidate-" ++ name, entry) | (name, entry) <- record (lookupRun "paired" "candidate" input) input] ++ scoreFlags input ++ probeFlags input)
    executed scheduled input = flags (record (lookupRun scheduled "candidate" input) input)
    encodeRow key pairedFlags repeats = toJSON (U.cohort key, U.task key, pairedFlags, repeats)
    collectedFor input = [value | value <- collected, U.inputKey (collectedInput value) == U.inputKey input]
    scoreFlags input = concat [scoreFlag value | value <- collectedFor input, collectedKind value == "score"]
    scoreFlag value =
        let prefix = case collectedPath value of
                N.Reference -> "reference-score-"
                N.Candidate -> "candidate-score-"
            V.Binding (V.CallId call) (V.AttemptId attempt) (V.Instance instance') = collectedBinding value
         in [ (prefix ++ "log", collectedLog value)
            , (prefix ++ "exit-code", show (collectedExit value))
            , (prefix ++ "call", show call)
            , (prefix ++ "attempt", show attempt)
            , (prefix ++ "instance", show instance')
            ]
    probeFlags input = case [value | value <- collectedFor input, collectedKind value == "probe"] of
        [] -> []
        values -> [("probe-set", Text.unpack (TextBytes.decodeUtf8 (Lazy.toStrict (encode (probeValue values)))))]
    probeValue values =
        [ object ["path" .= pathName path, "steps" .= collectedSteps value, "reference" .= endpoint N.Reference, "candidate" .= endpoint N.Candidate]
        | let endpoint target = maybe Null endpointValue (findTarget target values)
        , (path, value) <- Map.toList (Map.fromList [(collectedPath value, value) | value <- values])
        ]
    endpointValue value =
        let V.Binding (V.CallId call) (V.AttemptId attempt) (V.Instance instance') = collectedBinding value
         in object ["log" .= collectedLog value, "exit-code" .= show (collectedExit value), "call" .= call, "attempt" .= attempt, "instance" .= instance']
    findTarget target values = case [value | value <- values, collectedTarget value == target] of
        [value] -> Just value
        _ -> Nothing

digest :: Bytes.ByteString -> String
digest = Artifact.hex . SHA256.hash

options :: [OptDescr (String, String)]
options = O.descriptions [("contract", "Contract written by invar use prepare"), ("execution", "Execution plan (invar-use-execution-v1)"), ("output", "Directory to create for the contract copy, logs and records; must not exist")]
