{-# LANGUAGE OverloadedStrings #-}

module UseRun (run) where

import Check qualified
import Control.Exception (SomeException, displayException, try)
import Control.Monad (unless)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value, encode, object, toJSON, (.=))
import Data.Bifunctor (first)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Foldable (toList)
import Data.List (genericLength, mapAccumL)
import Data.Map.Strict qualified as Map
import Envelope (Problem (..), command, refuse, succeed)
import Envelope qualified
import InferenceInput qualified
import Invar.Artifact qualified as Artifact
import Invar.Infer.Observation qualified as Inference
import Invar.Infer.Replay qualified as Replay
import Invar.Infer.Session qualified as Session
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Numerical qualified as N
import Invar.Policy qualified as Policy
import Invar.Spec.Invocation qualified as V
import Invar.Use qualified as U
import Invar.Use.Execution qualified as E
import Invar.Use.Prepare qualified as Prepare
import Numeric.Natural (Natural)
import Options qualified as O
import System.Console.GetOpt (OptDescr)
import System.Directory (createDirectory, doesDirectoryExist, doesFileExist, makeAbsolute)
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
        decoded = Execution.decode executionValue `Check.andThen` \plan -> plan <$ Execution.schedules (Prepare.inputCount report) (Prepare.candidateExecutions report) plan
        refused = invalid ++ missing ++ unsupported contractPath contract
    declared <- either (refuse format . (refused ++)) pure (Check.run decoded)
    unless (null refused) (refuse format refused)
    plan <- resolved (takeDirectory executionPath) declared
    artifacts contract plan >>= \problems -> unless (null problems) (refuse format problems)
    let planned = groups contract plan
    createDirectory output
    createDirectory (output </> "logs")
    Bytes.writeFile (output </> "contract.json") contractBytes
    Lazy.writeFile (output </> "plan.json") (encode (object ["format" .= ("invar-use-run-plan-v1" :: String), "execution_sha256" .= digest executionBytes, "groups" .= map describe planned]))
    executable <- getExecutablePath
    records <- execute executable output plan planned
    Lazy.writeFile (output </> "runs.json") (encode (rows contract plan planned))
    succeed
        format
        "recorded"
        "Every record the contract needs under this execution plan was produced and retained."
        "That the candidate is admitted."
        [ "output" .= output
        , "contract_sha256" .= digest contractBytes
        , "execution_sha256" .= digest executionBytes
        , "groups" .= length records
        , "files" .= (["contract.json", "plan.json", "records.json", "runs.json"] :: [String])
        , "next" .= ("cd " ++ output ++ " && invar use admit --contract contract.json --runs runs.json")
        ]
  where
    required fields name = maybe (refuse format [Problem "missing-argument" ("argv:--" ++ name) ("--" ++ name ++ " is required")]) pure (O.optional fields name)

unsupported :: FilePath -> U.UseContract -> [Problem]
unsupported path contract =
    [ Problem "unsupported" ("artifact:" ++ path ++ ":criterion.numerical[" ++ show index ++ "]") "This requirement needs scored paths or full-vocabulary probes, which a run does not collect"
    | (index, requirement) <- zip [0 :: Int ..] (U.numericalRequirements (U.criterion contract))
    , scored (U.relation requirement) || not (null (U.probeSteps requirement))
    ]
  where
    scored relation = case relation of
        N.ScoredPathLogRatioWithin _ _ -> True
        N.FullVocabularyKLWithin {} -> True
        _ -> False

resolved :: FilePath -> Execution.Execution -> IO Execution.Execution
resolved base plan = do
    python <- absolute (Execution.python plan)
    cache <- absolute (Execution.cache plan)
    reference <- side (Execution.reference plan)
    candidate <- side (Execution.candidate plan)
    pure plan {Execution.python = python, Execution.cache = cache, Execution.reference = reference, Execution.candidate = candidate}
  where
    absolute path = makeAbsolute (base </> path)
    side selected = Execution.Side <$> absolute (Execution.worker selected) <*> absolute (Execution.configuration selected) <*> absolute (Execution.adapter selected)

artifacts :: U.UseContract -> Execution.Execution -> IO [Problem]
artifacts contract plan = do
    python <- file "execution:python" (Execution.python plan)
    cache <- directory "execution:cache" (Execution.cache plan)
    reference <- side "reference" (Execution.reference plan) (U.referenceImplementation contract)
    candidate <- side "candidate" (Execution.candidate plan) (U.candidateImplementation contract)
    pure (python ++ cache ++ reference ++ candidate)
  where
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

execute :: FilePath -> FilePath -> Execution.Execution -> [Group] -> IO [Value]
execute executable output plan = go []
  where
    go done [] = pure done
    go done (group : rest) = do
        (outcome, record) <- once group
        let kept = done ++ [record]
        Lazy.writeFile (output </> "records.json") (encode kept)
        either (\found -> refuse format [found {message = message found ++ "; every record so far is kept in " ++ (output </> "records.json")}]) (const (go kept rest)) outcome
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

checked :: Group -> (Bytes.ByteString, Bytes.ByteString) -> Int -> Either Problem Int
checked group (callBytes, logBytes) status = do
    declared <- first (Problem "internal-error" (located group) . ("The group's calls could not be read back: " ++)) (InferenceInput.calls callBytes)
    admitted <- first (Problem "execution-failed" (located group) . (("The log of group " ++ label group ++ " does not carry its declared batch: ") ++) . show) (Replay.standalone Session.Batched (Session.Declaration declared Nothing) (Replay.declared status) logBytes)
    mapM_ (member admitted) (members group)
    pure status
  where
    member admitted (input, call) = do
        let binding = V.Binding (V.CallId call) (V.AttemptId call) (V.Instance call)
            named = label group ++ " input " ++ show (U.cohort (U.inputKey input), U.task (U.inputKey input))
        report <- case filter ((== binding) . Trajectory.binding . Replay.trajectory) admitted of
            [single] -> pure (Inference.view single)
            _ -> Left (Problem "execution-failed" (located group) ("The log does not carry the declared observation for " ++ named))
        reported <- first (Problem "identity-mismatch" (located group) . (("The worker's report for " ++ named ++ " names no policy: ") ++)) (Inference.policyDescription report)
        unless (reported == policy group) (Left (Problem "identity-mismatch" (located group) ("The worker reported " ++ show reported ++ " for " ++ named ++ ", not the contract's " ++ sideName group ++ " policy")))

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

rows :: U.UseContract -> Execution.Execution -> [Group] -> [Value]
rows contract plan planned =
    [ encodeRow (U.inputKey input) (paired input) [executed ("repeats[" ++ show index ++ "]") input | index <- [0 .. length (Execution.repeats plan) - 1]]
    | input <- toList (U.declaredInputs (U.declaredDomain contract))
    ]
  where
    placed = Map.fromList [((schedule group, sideName group, U.inputKey input), (group, call)) | group <- planned, (input, call) <- members group]
    lookupRun scheduled owner input = placed Map.! (scheduled, owner, U.inputKey input)
    record (group, call) input = request (policy group) input call ++ [("log", logPath group), ("calls", callsPath group), ("exit-code", "0")]
    paired input = flags ([("reference-" ++ name, entry) | (name, entry) <- record (lookupRun "paired" "reference" input) input] ++ [("candidate-" ++ name, entry) | (name, entry) <- record (lookupRun "paired" "candidate" input) input])
    executed scheduled input = flags (record (lookupRun scheduled "candidate" input) input)
    encodeRow key pairedFlags repeats = toJSON (U.cohort key, U.task key, pairedFlags, repeats)

digest :: Bytes.ByteString -> String
digest = Artifact.hex . SHA256.hash

options :: [OptDescr (String, String)]
options = O.descriptions [("contract", "Contract written by invar use prepare"), ("execution", "Execution plan (invar-use-execution-v1)"), ("output", "Directory to create for the contract copy, logs and records; must not exist")]
