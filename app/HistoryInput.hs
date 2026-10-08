module HistoryInput (trace, initialPolicy, traceOptions, inspect, inspectInitial, compareHistories, options, inputOptions, load, select) where

import Control.Monad (unless)
import Data.Aeson (encode)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Data.List (isPrefixOf)
import Data.Map.Strict qualified as Map
import Data.Maybe (isNothing)
import InferenceInput qualified
import Invar.History qualified as History
import Invar.History.Initial qualified as Initial
import Invar.History.Trace qualified as Trace
import Invar.Policy qualified as Policy
import Invar.Workload qualified as Workload
import NativeCodec qualified
import Options qualified as O
import System.Console.GetOpt (OptDescr (Option), usageInfo)
import System.Exit (die)
import System.FilePath ((</>))
import Training qualified

trace :: O.Fields -> IO (Trace.Run, Workload.Document, ByteString)
trace fields = do
    settings <- either die pure (Training.settings fields)
    path <- either die pure (O.required fields "tasks")
    tasks <- Bytes.readFile path >>= either die pure . Workload.decode
    sessions <- either die pure (O.numeric fields "sessions")
    output <- either die pure (O.required fields "output")
    method <- either die pure (O.required fields "publication")
    status <- either die pure (O.numeric fields "exit-code")
    logPath <- either die pure (O.required fields "log")
    encoded <- Bytes.readFile logPath
    inference <- either die pure (lifetimeMode fields ("inference-mode", [("serial", Trace.Finite), ("batch", Trace.Batched)]))
    learning <- either die pure (lifetimeMode fields ("learning-mode", [("process", Trace.Finite)]))
    pure (Trace.Run {Trace.settings = settings, Trace.sessions = sessions, Trace.output = output, Trace.method = method, Trace.exitCode = status, Trace.inferenceMode = inference, Trace.learningMode = learning}, tasks, encoded)

initialPolicy :: O.Fields -> IO Policy.Description
initialPolicy fields = do
    checkpoint <- either die pure (O.required fields "checkpoint")
    Policy.readDescription (checkpoint </> "policy.json")

lifetimeMode :: O.Fields -> (String, [(String, Trace.Mode)]) -> Either String Trace.Mode
lifetimeMode fields (key, finite) = case O.optional fields key of
    Nothing -> Right Trace.Finite
    Just "resident" -> Right Trace.Resident
    Just "shared" -> Right Trace.Shared
    Just value | Just selected <- lookup value finite -> Right selected
    _ -> Left ("Unsupported declared --" ++ key)

inspect :: [String] -> IO ()
inspect ["--help"] = putStrLn (usageInfo "Usage: invar inspect history OPTIONS" options)
inspect supplied = do
    fields <- either die pure (O.parse options supplied)
    decoder <- either die pure (NativeCodec.select fields)
    (declared, output) <- load fields
    observed <- History.admit decoder declared output
    Lazy.putStrLn (encode (History.describe observed))

load :: O.Fields -> IO (History.Declaration, ByteString)
load fields = do
    training <- case O.optional fields "run" of
        Just directory -> do
            unless (null [name | name <- names traceOptions ++ ["reference"], Map.member name fields]) (die "--run takes the training arguments and workload from the run's declaration")
            pure (History.Recorded directory Training.declared)
        Nothing -> do
            (run, tasks, encoded) <- trace fields
            checkpoint <- either die pure (O.required fields "checkpoint")
            reference <- either die pure (O.required fields "reference")
            pure (History.Logged (History.Log run tasks checkpoint reference encoded))
    finalRequest <- either die pure (InferenceInput.requestWith "final-" fields)
    finalBinding <- either die pure (InferenceInput.bindingWith "final-" fields)
    finalExit <- either die pure (O.numeric fields "final-exit-code")
    finalPath <- either die pure (O.required fields "final-log")
    finalOutput <- Bytes.readFile finalPath
    random <- either die pure (randomProfile fields)
    source <- initialSource fields
    mode <- either die pure $ case O.optional fields "profile-mode" of
        Just "unreported" -> Right History.Unreported
        Just "uniform" -> Right History.Uniform
        Just "roles" -> Right History.Roles
        _ -> Left "Expected explicit --profile-mode unreported, uniform or roles"
    let declared = History.Declaration {History.training = training, History.randomProfile = random, History.initialSource = source, History.profileMode = mode, History.finalRequest = finalRequest, History.finalBinding = finalBinding, History.finalExit = finalExit}
    pure (declared, finalOutput)

names :: [OptDescr (String, String)] -> [String]
names described = concat [long | Option _ long _ _ <- described]

initialSource :: O.Fields -> IO Initial.Source
initialSource fields = case O.optional fields "initial-source" of
    Just "provided" -> do
        unless (all (isNothing . O.optional fields) ["initial-log", "initial-exit-code", "initial-seed"]) (die "Provided initial checkpoint does not accept initializer observations")
        pure Initial.Provided
    Just "initializer" -> do
        path <- either die pure (O.required fields "initial-log")
        status <- either die pure (O.numeric fields "initial-exit-code")
        seed <- either die pure (O.numeric fields "initial-seed")
        Initial.Executed (Initial.Run seed status) <$> Bytes.readFile path
    _ -> die "Expected explicit --initial-source provided or initializer"

inspectInitial :: [String] -> IO ()
inspectInitial ["--help"] = putStrLn (usageInfo "Usage: invar inspect initial OPTIONS" initialOptions)
inspectInitial supplied = do
    fields <- either die pure (O.parse initialOptions supplied)
    decoder <- either die pure (NativeCodec.select fields)
    settings <- either die pure (Training.settings fields)
    checkpoint <- either die pure (O.required fields "checkpoint")
    random <- either die pure (randomProfile fields)
    source <- initialSource fields
    observed <- Initial.admit decoder (settings, checkpoint, random) source
    Lazy.putStrLn (encode (Initial.describe observed))

randomProfile :: O.Fields -> Either String Initial.Random
randomProfile fields = case O.optional fields "rng-profile" of
    Nothing -> torch
    Just "torch" -> torch
    Just "mlx" -> do
        unless (isNothing (O.optional fields "cuda-rng-vectors")) (Left "MLX RNG does not accept a CUDA vector declaration")
        pure Initial.MLX
    _ -> Left "Expected --rng-profile torch or mlx"
  where
    torch = Initial.Torch <$> O.numeric fields "cuda-rng-vectors"

compareHistories :: [String] -> IO ()
compareHistories ["--help"] = putStrLn (usageInfo "Usage: invar compare histories OPTIONS" pairOptions)
compareHistories supplied = do
    fields <- either die pure (O.parse pairOptions supplied)
    decoder <- either die pure (NativeCodec.select fields)
    (first, firstOutput) <- load (select "left-" fields)
    (second, secondOutput) <- load (select "right-" fields)
    left <- History.admit decoder first firstOutput
    right <- History.admit decoder second secondOutput
    History.compare (left, right) >>= Lazy.putStrLn . encode

select :: String -> O.Fields -> O.Fields
select prefix fields = Map.fromList [(drop (length prefix) key, value) | (key, value) <- Map.toList fields, prefix `isPrefixOf` key]

pairOptions :: [OptDescr (String, String)]
pairOptions = NativeCodec.options ++ concatMap (`O.prefixed` inputOptions) ["left-", "right-"]

traceOptions :: [OptDescr (String, String)]
traceOptions = Training.settingsOptions ++ O.descriptions [("checkpoint", "Complete initial checkpoint directory, whose policy.json declares the first generation"), ("tasks", "Frozen workload file"), ("log", "Complete training stdout"), ("sessions", "Declared physical inference owner count"), ("inference-mode", "Declared serial/batch (finite default), resident or shared inference lifetime"), ("learning-mode", "Declared process (default), resident or shared learner lifetime"), ("output", "Declared training output directory"), ("publication", "Declared rename or reference publication method"), ("exit-code", "Independently observed training process exit status")]

options :: [OptDescr (String, String)]
options = inputOptions ++ NativeCodec.options

inputOptions :: [OptDescr (String, String)]
inputOptions = traceOptions ++ InferenceInput.optionsWith "final-" ++ initialInputOptions ++ O.descriptions [("run", "Output directory of a run of the event runtime; its journal declares the training arguments and workload, which replace the training log options"), ("reference", "Fixed reference adapter file"), ("profile-mode", "unreported, uniform across all processes, or roles with complete profiles uniform within inference, learning and shared owners separately"), ("final-log", "Complete independent final inference stdout"), ("final-exit-code", "Independently observed final inference process exit status")]

initialInputOptions :: [OptDescr (String, String)]
initialInputOptions = O.descriptions [("rng-profile", "torch (default) or native mlx"), ("cuda-rng-vectors", "Declared CUDA RNG vector count in initial and successor learners"), ("initial-source", "provided or initializer"), ("initial-log", "Complete actual initializer stdout"), ("initial-exit-code", "Independently observed initializer process exit status"), ("initial-seed", "Declared initializer seed")]

initialOptions :: [OptDescr (String, String)]
initialOptions = Training.settingsOptions ++ O.descriptions [("checkpoint", "Complete initial checkpoint directory")] ++ initialInputOptions ++ NativeCodec.options
