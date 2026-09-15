{-# LANGUAGE OverloadedStrings #-}

module Updates (updates, checkpointResult, setup, setupFor, field, change, alter, wire, observe) where

import Control.Monad (forM_, void)
import Data.Aeson (Value (..), encode, object, toJSON, (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Foldable (toList)
import Data.Maybe (fromMaybe, listToMaybe)
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8)
import Hedgehog
import Invar.Learn.Program qualified as Program
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Wire qualified as Wire
import Invar.Learn.Worker qualified as Worker
import Invar.Qualification qualified as Gate
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as Load
import Learning (world)
import Store (workspace)
import System.Directory (createDirectory, doesFileExist)
import System.FilePath ((</>))
import System.IO.Error (ioeGetErrorString, isDoesNotExistError, tryIOError)
import System.Posix.Files (createSymbolicLink)

type Context = (V.Binding, V.Runtime)

updates :: Group
updates = Group "Bound update reports" [("completion retains the logical emission and actual wire request", once completed), ("each phase binds all invocation identities", once bindings), ("loaded state and actual inputs must match the checked lowering", once inputs), ("program identity and report order remain mandatory", once protocol), ("staged result must match the update and admitted token count", once result), ("gradient observation identity is mandatory", once gradients), ("gradient verification reads the complete reported artifact", once gradientFile), ("update approval requires actual loaded and consumed inputs", once approval), ("learner load identity and consumed lifetime authorize the update", once loading), ("completed updates extend the accepted consumption without renewed authority", once ownership)]
  where
    once = withTests 1 . property

binding :: V.Binding
binding = V.Binding (V.CallId 7) (V.AttemptId 11) (V.Instance 13)

bound :: Value
bound = object ["call" .= Number 7, "attempt" .= Number 11, "instance" .= Number 13]

setup :: PropertyT IO (Context, [Value])
setup = setupFor world

setupFor :: E.World -> PropertyT IO (Context, [Value])
setupFor supplied = do
    checked <- evalEither Program.checked
    prepared <- evalEither (V.prepare (V.Selection (V.boundCall binding) supplied) (V.start checked 0))
    let runtime = prepared
    command <- evalEither (V.intent runtime (V.boundCall binding))
    actual <- evalEither (Wire.lower command)
    loadProgram <- evalEither (P.loadProgram (binding, runtime))
    image <- evalEither (Wire.image command)
    let load = object ["binding" .= bound, "program" .= decodeUtf8 loadProgram]
        imageValue = object ["artifact" .= decodeUtf8 (Load.artifact image), "profile" .= decodeUtf8 (Load.profile image)]
        state = object ["policy" .= field "policy" actual, "learner" .= field "learner" actual, "tokenizer" .= field "tokenizer" actual, "base" .= field "base" actual, "assembly" .= field "assembly" actual, "reference" .= field "reference" actual, "optimizer" .= field "optimizer" actual]
        loaded = object ["stage" .= String "loaded_learner", "binding" .= bound, "state" .= state, "load" .= load, "image" .= imageValue]
        consumed = object ["stage" .= String "consumed", "binding" .= bound, "program" .= decodeUtf8 (A.bytes checked), "request" .= actual, "load" .= load]
        update = object ["before" .= field "policy" actual, "after" .= digest "d", "active_tokens" .= Number 2, "nonzero_advantages" .= Number 2, "loss" .= Number 0, "gradient_norm" .= Number 1, "reward_gradient_norm" .= Number 1]
        finished = object ["stage" .= String "result", "binding" .= bound, "request" .= actual, "update" .= update, "adapter" .= digest "d", "learner" .= digest "e", "gradients" .= digest "f", "probabilities" .= digest "a", "storage" .= String "staged; not published"]
    pure ((binding, runtime), [loaded, consumed, finished])

checkpointResult :: String -> String -> PropertyT IO P.Result
checkpointResult policy learner = do
    (context, events) <- setup
    let adapter = String (Text.pack policy)
        finished value = change "adapter" adapter (change "learner" (String (Text.pack learner)) (change "update" (change "after" adapter (field "update" value)) value))
    evalEither (observe context (wire (alter 2 finished events)))

digest :: Text.Text -> Value
digest = String . Text.replicate digestLength
  where
    digestLength = 64

field :: Key -> Value -> Value
field name (Object fields) = fromMaybe (error "Missing update fixture field") (Fields.lookup name fields)
field _ _ = error "Expected an update fixture object"

change :: Key -> Value -> Value -> Value
change name value (Object fields) = Object (Fields.insert name value fields)
change _ _ _ = error "Expected an update fixture object"

alter :: Int -> (value -> value) -> [value] -> [value]
alter selected transform = zipWith (\index value -> if index == selected then transform value else value) [0 ..]

wire :: [Value] -> ByteString
wire = Bytes.unlines . map (Lazy.toStrict . encode)

observe :: Context -> ByteString -> Either P.Error P.Result
observe context output = do
    let prefix = Bytes.unlines (takeThroughConsumed (Bytes.lines output))
    (_, permit) <- P.authorize Gate.empty context prefix
    P.observe permit output
  where
    takeThroughConsumed [] = []
    takeThroughConsumed (line : remaining)
        | "\"stage\":\"consumed\"" `Bytes.isInfixOf` line = [line]
        | otherwise = line : takeThroughConsumed remaining

completed :: PropertyT IO ()
completed = do
    (context@(_, runtime), events) <- setup
    actual <- evalEither (observe context (wire events))
    command <- evalEither (V.intent runtime (V.boundCall binding))
    V.completedBinding (P.completion actual) === binding
    V.completedEmission (P.completion actual) === command
    V.completedOutput (P.completion actual) === Lazy.toStrict (encode (last events))
    P.request actual === field "request" (events !! 1)
    P.adapter actual === replicate 64 'd'
    P.learner actual === replicate 64 'e'
    P.gradients actual === replicate 64 'f'

bindings :: PropertyT IO ()
bindings = do
    (context, events) <- setup
    forM_ [0 .. length events - 1] $ \position ->
        forM_ ["call", "attempt", "instance"] $ \name -> do
            let changed = alter position (change "binding" (change name (Number 99) bound)) events
            case observe context (wire changed) of
                Left (P.Lifecycle (V.BindingMismatch expected actual)) -> expected === binding >> assert (actual /= binding)
                unexpected -> annotateShow unexpected >> failure

inputs :: PropertyT IO ()
inputs = do
    (context, events) <- setup
    first <- evalMaybe (listToMaybe events)
    let actual = field "request" (events !! 1)
        loaded = field "state" first
    forM_ ["policy", "learner", "tokenizer", "base", "assembly", "reference", "optimizer"] $ \name ->
        mismatch (observe context (wire (alter 0 (change "state" (change name Null loaded)) events)))
    forM_ [1, 2] $ \position ->
        forM_ ["specification", "policy", "learner", "tokenizer", "base", "assembly", "behavior_model", "reference", "samples", "order", "epsilon", "penalty", "delta", "optimizer"] $ \name ->
            mismatch (observe context (wire (alter position (change "request" (change name Null actual)) events)))
    forM_ ["base", "assembly"] $ \name -> do
        let changed = change "behavior_model" (change name (digest "2") (field "behavior_model" actual)) actual
            history = alter 2 (change "request" changed) (alter 1 (change "request" changed) events)
        mismatch (observe context (wire history))
    case field "samples" actual of
        Array delivered -> do
            let changed name value = change "samples" (toJSON (alter 0 (change name value) (toList delivered))) actual
                negativeZero = 2147483648
                mutations = [("advantage_bits", Number 0), ("advantage_bits", Number negativeZero), ("reward", Number 1), ("group", String "other")]
            forM_ mutations $ \(name, value) -> do
                let reported = changed name value
                    history = alter 2 (change "request" reported) (alter 1 (change "request" reported) events)
                mismatch (observe context (wire history))
        _ -> failure

protocol :: PropertyT IO ()
protocol = do
    (context, events) <- setup
    observe context (wire (alter 1 (change "program" (String "other")) events)) === Left (P.Lifecycle V.ProgramMismatch)
    case events of
        [loaded, consumed, finished] ->
            forM_ [[], [loaded], [loaded, finished], [consumed, loaded, finished], [loaded, loaded, consumed, finished], [loaded, consumed, consumed, finished], [loaded, consumed, finished, finished], events ++ [object ["stage" .= String "reward_update"]]] $ \history ->
                case observe context (wire history) of
                    Left (P.Unexpected _) -> success
                    unexpected -> annotateShow unexpected >> failure
        _ -> failure

result :: PropertyT IO ()
result = do
    (context, events) <- setup
    let update = field "update" (last events)
        changes = [("before", digest "f"), ("after", digest "f"), ("active_tokens", Number 3), ("nonzero_advantages", Number 3), ("nonzero_advantages", Number (-1)), ("gradient_norm", Number (-1)), ("reward_gradient_norm", Number (-1)), ("loss", Number (10 ^ overflowExponent))]
    forM_ changes $ \(name, value) ->
        mismatch (observe context (wire (alter 2 (change "update" (change name value update)) events)))
    forM_ [("adapter", digest "f"), ("learner", String "unknown"), ("storage", String "published")] $ \(name, value) ->
        mismatch (observe context (wire (alter 2 (change name value) events)))
  where
    overflowExponent = 400 :: Int

gradients :: PropertyT IO ()
gradients = do
    (context, events) <- setup
    let omit (Object fields) = Object (Fields.delete "gradients" fields)
        omit value = value
    case observe context (wire (alter 2 omit events)) of
        Left (P.Malformed _) -> success
        unexpected -> annotateShow unexpected >> failure
    forM_ [String "unknown", String (Text.replicate 64 "F"), String ""] $ \invalid ->
        mismatch (observe context (wire (alter 2 (change "gradients" invalid) events)))

gradientFile :: PropertyT IO ()
gradientFile = do
    (context, events) <- setup
    let expected = "0ddd6b7433742e7920ec0337e33d65cb6cd53541d7ab48eeb306d17f39a37cce"
        content = Bytes.replicate payloadLength 'x' <> "y"
        payloadLength = 70000
    observed <- evalEither (observe context (wire (alter 2 (change "gradients" (String (Text.pack expected))) events)))
    root <- workspace
    let path = root </> "gradients.safetensors"
    missing <- evalIO (tryIOError (Worker.verifyGradients root observed))
    case missing of
        Left problem -> assert (isDoesNotExistError problem)
        Right unexpected -> annotateShow unexpected >> failure
    evalIO (Bytes.writeFile path content)
    evalIO (Worker.verifyGradients root observed) >>= (=== Right ())
    evalIO (Bytes.appendFile path "changed")
    changed <- evalIO (Worker.verifyGradients root observed)
    case changed of
        Left (Worker.GradientMismatch claimed actual) -> claimed === expected >> assert (actual /= expected)
        unexpected -> annotateShow unexpected >> failure
    rejectedPaths observed path

rejectedPaths :: P.Result -> FilePath -> PropertyT IO ()
rejectedPaths observed target = do
    directory <- workspace
    evalIO (createDirectory (directory </> "gradients.safetensors"))
    folder <- evalIO (tryIOError (Worker.verifyGradients directory observed))
    case folder of
        Left problem -> ioeGetErrorString problem === "Gradient observation is not a regular file"
        Right unexpected -> annotateShow unexpected >> failure
    linked <- workspace
    evalIO (createSymbolicLink target (linked </> "gradients.safetensors"))
    evalIO (doesFileExist (linked </> "gradients.safetensors")) >>= assert
    symbolic <- evalIO (tryIOError (Worker.verifyGradients linked observed))
    case symbolic of
        Left _ -> success
        Right unexpected -> annotateShow unexpected >> failure

mismatch :: Either P.Error P.Result -> PropertyT IO ()
mismatch value = case value of
    Left (P.Mismatch _) -> success
    unexpected -> annotateShow unexpected >> failure

approval :: PropertyT IO ()
approval = do
    (context, events) <- setup
    let pending = take 2 events
        actual = field "request" (events !! 1)
    void (P.authorize Gate.empty context (wire pending)) === Right ()
    forM_ [[], take 1 events, events, reverse pending] $ \history ->
        case P.authorize Gate.empty context (wire history) of
            Left _ -> success
            Right _ -> failure
    forM_ ["policy", "learner", "tokenizer", "base", "assembly", "reference", "samples", "order", "optimizer"] $ \name ->
        case void (P.authorize Gate.empty context (wire (alter 1 (change "request" (change name Null actual)) pending))) of
            Left (P.Mismatch _) -> success
            unexpected -> annotateShow unexpected >> failure

loading :: PropertyT IO ()
loading = do
    (context, events) <- setup
    first <- evalMaybe (listToMaybe events)
    let pending = take 2 events
        loadEnvelope = field "load" first
        image = field "image" first
        changes =
            [alter 0 (change "image" (change name (digest "0") image)) pending | name <- ["artifact", "profile"]]
                ++ [alter position (change "load" (change "program" (String "different") loadEnvelope)) pending | position <- [0, 1]]
                ++ [alter position (change "load" (change "binding" (change name (Number 99) bound) loadEnvelope)) pending | position <- [0, 1], name <- ["call", "attempt", "instance"]]
    forM_ changes $ \history ->
        case P.authorize Gate.empty context (wire history) of
            Left _ -> success
            Right _ -> failure

ownership :: PropertyT IO ()
ownership = do
    (context, events) <- setup
    (registry, permit) <- evalEither (P.authorize Gate.empty context (wire (take 2 events)))
    let fact = P.loadedFact permit
        name = V.boundInstance binding
        closed = Gate.close registry
    Load.active (Gate.loads registry) === [name]
    Load.active (Gate.loads closed) === []
    Load.historical (Gate.loads closed) name === Right fact
    V.completedBinding (Load.report fact) === binding
    observed <- evalEither (P.observe permit (wire events))
    assert (V.completedProgram (Load.report fact) /= V.completedProgram (P.completion observed))
    forM_ [registry, closed] $ \previous ->
        case P.authorize previous context (wire (take 2 events)) of
            Left _ -> success
            Right _ -> failure
    mismatch (P.observe permit (wire (alter 0 (change "image" Null) events)))
