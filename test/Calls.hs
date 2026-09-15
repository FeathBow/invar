{-# LANGUAGE OverloadedStrings #-}

module Calls (calls, setup, request, change, field, wire, reviewPrefix, permissionInput, observe) where

import Control.Monad (forM_)
import Data.Aeson (Value (..), eitherDecodeStrict, encode, object, (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Maybe (fromMaybe)
import Data.Text qualified as Text
import Data.Text.Encoding (encodeUtf8)
import Data.Word (Word32)
import Hedgehog
import Invar.Infer qualified as I
import Invar.Infer.Invocation qualified as C
import Invar.Infer.Result qualified as R
import Invar.Qualification qualified as Gate
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as L
import Invar.Worker qualified as W
import Store (workspace)
import System.Directory (doesFileExist)
import System.Exit (ExitCode (..))
import System.FilePath ((</>))

calls :: Group
calls = Group "Bound inference protocol" [("completed reports retain the producing invocation", once completed), ("each phase binds call attempt and instance", once bindings), ("consumption matches exact program and semantic arguments", once consumed), ("missing repeated and reordered consumption cannot complete", once protocol), ("approval requires matching load and consumption before completion", once approval), ("worker transport withholds permission for mismatched consumption", once handshake), ("load facts and reviewed prefixes remain distinct from permission", once loadOwnership)]
  where
    once = withTests 1 . property

bound :: V.Binding
bound = V.Binding (V.CallId 7) (V.AttemptId 11) (V.Instance 13)

binding :: Value
binding = object ["call" .= Number 7, "attempt" .= Number 11, "instance" .= Number 13]

identity :: String
identity = replicate 64 'a'

request :: I.Request
request = I.Request identity (replicate 64 'c') (replicate 64 'e') (replicate 64 'f') "Compute the answer." 2 0.8 17

requestValue :: Value
requestValue = object ["prompt" .= String "Compute the answer.", "tokens" .= Number 2, "temperature" .= Number 0.8, "seed" .= Number 17]

setup :: PropertyT IO (C.Call, [Value])
setup = do
    planned <- evalEither (I.prepare request)
    call <- evalEither (C.prepare bound planned)
    envelope <- evalEither (eitherDecodeStrict (encodeUtf8 (Text.pack (C.input call))))
    let program = field "program" envelope
        loading = field "load" envelope
        image = I.image request
        loaded = object ["stage" .= String "loaded_adapter", "binding" .= binding, "load" .= loading, "image" .= object ["artifact" .= Bytes.unpack (L.artifact image), "profile" .= Bytes.unpack (L.profile image)], "requested" .= identity, "consumed" .= identity, "tokenizer" .= replicate 64 'c', "base" .= replicate 64 'e', "assembly" .= replicate 64 'f']
        consumption = object ["stage" .= String "consumed", "binding" .= binding, "program" .= program, "load" .= loading, "adapter" .= identity, "tokenizer" .= replicate 64 'c', "base" .= replicate 64 'e', "assembly" .= replicate 64 'f', "request" .= requestValue]
        result = object ["stage" .= String "result", "binding" .= binding, "adapter" .= identity, "tokenizer" .= replicate 64 'c', "base" .= replicate 64 'e', "assembly" .= replicate 64 'f', "request" .= requestValue, "tokens" .= [1, 2, 3 :: Int], "prompt_length" .= Number 1, "behavior" .= [-0.5, -0.25 :: Double], "behavior_bits" .= [0xbf000000, 0xbe800000 :: Word32], "text" .= String "#### 12", "truncated" .= True]
    field "binding" envelope === binding
    pure (call, [loaded, consumption, result])

field :: Key -> Value -> Value
field name (Object fields) = fromMaybe (error "Missing protocol fixture field") (Fields.lookup name fields)
field _ _ = error "Protocol fixture must be an object"

change :: Key -> Value -> Value -> Value
change name value (Object fields) = Object (Fields.insert name value fields)
change _ _ _ = error "Protocol fixture must be an object"

wire :: [Value] -> ByteString
wire = Bytes.unlines . map (Lazy.toStrict . encode)

reviewPrefix :: [Value] -> [Value]
reviewPrefix [] = []
reviewPrefix (event : remaining)
    | field "stage" event == String "consumed" = [event]
    | otherwise = event : reviewPrefix remaining

permissionInput :: C.Call -> ByteString
permissionInput call = case eitherDecodeStrict (encodeUtf8 (Text.pack (C.input call))) of
    Right value -> Lazy.toStrict (encode (object ["binding" .= field "binding" value, "program" .= field "program" value]))
    Left problem -> error problem

observe :: C.Call -> ByteString -> Either C.Error (V.Completion, R.Result)
observe call encoded = do
    events <- either (Left . C.Protocol) Right (traverse eitherDecodeStrict (Bytes.lines encoded))
    (_, permit) <- C.authorize Gate.empty call (wire (reviewPrefix events))
    C.observe permit encoded

completed :: PropertyT IO ()
completed = do
    (call, events) <- setup
    (completion, result) <- evalEither (observe call (wire events))
    V.completedBinding completion === bound
    V.completedCommand completion === 0
    V.completedOutput completion === Lazy.toStrict (encode (last events))
    case field "program" (events !! 1) of
        String text -> V.completedProgram completion === encodeUtf8 text
        _ -> failure
    R.consumed result === request
    R.response result === "#### 12"

bindings :: PropertyT IO ()
bindings = do
    (call, events) <- setup
    let cases = [("call", bound {V.boundCall = V.CallId 99}), ("attempt", bound {V.boundAttempt = V.AttemptId 99}), ("instance", bound {V.boundInstance = V.Instance 99})]
    forM_ [0 .. length events - 1] $ \position ->
        forM_ cases $ \(name, expected) -> do
            let changed = alter position (change "binding" (change name (Number 99) binding)) events
            lifecycle (V.BindingMismatch bound expected) (observe call (wire changed))

consumed :: PropertyT IO ()
consumed = do
    (call, events) <- setup
    lifecycle V.ProgramMismatch (observe call (wire (alter 1 (change "program" (String "different program")) events)))
    forM_ [("prompt", String "other"), ("tokens", Number 3), ("temperature", Number 1), ("seed", Number 18)] $ \(name, value) ->
        emissionMismatch (observe call (wire (alter 1 (change "request" (change name value requestValue)) events)))
    emissionMismatch (observe call (wire (alter 1 (change "adapter" (String (Text.replicate 64 "b"))) events)))
    emissionMismatch (observe call (wire (alter 1 (change "tokenizer" (String (Text.replicate 64 "b"))) events)))

protocol :: PropertyT IO ()
protocol = do
    (call, events) <- setup
    case events of
        [loaded, consumption, result] -> do
            case observe call (wire [loaded, result]) of
                Left (C.Result (R.Unexpected _)) -> success
                _ -> failure
            lifecycle (V.PhaseMismatch V.Issued V.Consumed) (observe call (wire [loaded, consumption, consumption, result]))
            case observe call (wire [consumption, loaded, result]) of
                Left (C.Result (R.Unexpected _)) -> success
                _ -> failure
        _ -> failure

alter :: Int -> (value -> value) -> [value] -> [value]
alter selected transform = zipWith (\index value -> if index == selected then transform value else value) [0 ..]

lifecycle :: V.Error -> Either C.Error value -> PropertyT IO ()
lifecycle expected result = case result of
    Left (C.Lifecycle actual) -> actual === expected
    Left unexpected -> annotateShow unexpected >> failure
    Right _ -> failure

emissionMismatch :: Either C.Error value -> PropertyT IO ()
emissionMismatch result = case result of
    Left (C.Lifecycle (V.EmissionMismatch _ _)) -> success
    Left unexpected -> annotateShow unexpected >> failure
    Right _ -> failure

approval :: PropertyT IO ()
approval = do
    (call, events) <- setup
    let pending = take 2 events
    (fmap (C.permission . snd) . C.authorize Gate.empty call) (wire pending) === Right (permissionInput call)
    forM_ [[], take 1 events, events, reverse pending] $ \history ->
        case (fmap (C.permission . snd) . C.authorize Gate.empty call) (wire history) of
            Left _ -> success
            Right _ -> failure
    lifecycle V.ProgramMismatch ((fmap (C.permission . snd) . C.authorize Gate.empty call) (wire (alter 1 (change "program" (String "other")) pending)))
    emissionMismatch ((fmap (C.permission . snd) . C.authorize Gate.empty call) (wire (alter 1 (change "adapter" (String (Text.replicate 64 "b"))) pending)))
    emissionMismatch ((fmap (C.permission . snd) . C.authorize Gate.empty call) (wire (alter 1 (change "tokenizer" (String (Text.replicate 64 "b"))) pending)))
    forM_ ["base", "assembly"] $ \name ->
        emissionMismatch ((fmap (C.permission . snd) . C.authorize Gate.empty call) (wire (alter 1 (change name (String (Text.replicate 64 "0"))) pending)))

handshake :: PropertyT IO ()
handshake = do
    (call, events) <- setup
    let pending = take 2 events
        changed = alter 1 (change "program" (String "other")) pending
        wrongLoad = alter 0 (change "tokenizer" (String (Text.replicate 64 "b"))) pending
        wrongConsumption = alter 1 (change "tokenizer" (String (Text.replicate 64 "b"))) pending
        wrongModels = [(False, alter index (change name (String (Text.replicate 64 "0"))) pending) | index <- [0, 1], name <- ["base", "assembly"]]
        wrongImages = [(False, alter 0 (\value -> change "image" (change name (String "wrong") (field "image" value)) value) pending) | name <- ["artifact", "profile"]]
        wrongPrograms = [(False, alter index (\value -> change "load" (change "program" (String "wrong") (field "load" value)) value) pending) | index <- [0, 1]]
    forM_ ([(True, pending), (False, changed), (False, wrongLoad), (False, wrongConsumption)] ++ wrongPrograms ++ wrongModels ++ wrongImages) $ \(valid, reported) -> do
        root <- workspace
        let script = root </> "handshake.sh"
            marker = root </> "approved"
            configuration = root </> "native config.json"
            body = unlines ["test \"$3\" = " ++ quote ("--config=" ++ configuration) ++ " || exit 20", "IFS= read -r invocation || exit 21", "printf '%s\\n' " ++ unwords (map (quote . Bytes.unpack) (Bytes.lines (wire reported))), "IFS= read -r permission || exit 22", "test \"$permission\" = " ++ quote (Bytes.unpack (permissionInput call)) ++ " || exit 23", "printf '%s' \"$permission\" > " ++ quote marker, "exit 7"]
        evalIO (writeFile script body)
        outcome <- evalIO (W.run (W.Worker "/bin/sh" script root root [] (Just configuration) Nothing) call)
        case (valid, outcome) of
            (True, Left (W.WorkerExit (ExitFailure 7))) -> success
            (False, Left (W.InvalidOutput _)) -> success
            _ -> failure
        evalIO (doesFileExist marker) >>= (=== valid)
  where
    quote text = "'" ++ concatMap (\character -> if character == '\'' then "'\\''" else [character]) text ++ "'"

loadOwnership :: PropertyT IO ()
loadOwnership = do
    (call, events) <- setup
    let prefix = wire (reviewPrefix events)
    (registry, permit) <- evalEither (C.authorize Gate.empty call prefix)
    let fact = C.loadFact permit
        closed = Gate.close registry
    L.active (Gate.loads registry) === [V.Instance 13]
    L.active (Gate.loads closed) === []
    L.historical (Gate.loads closed) (V.Instance 13) === Right fact
    V.completedEmission (L.report fact) === L.expectedEmission (I.image request)
    assert (V.completedProgram (L.report fact) /= encodeUtf8 (textField "program" (events !! 1)))
    _ <- evalEither (C.observe permit (wire events))
    forM_ [registry, closed] $ \state -> case C.authorize state call prefix of
        Left _ -> success
        Right _ -> failure
    case C.observe permit (wire (alter 0 (change "requested" (String "other")) events)) of
        Left (C.Protocol _) -> success
        _ -> failure
  where
    textField name value = case field name value of
        String text -> text
        _ -> error "Expected textual fixture field"
