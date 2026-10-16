{-# LANGUAGE OverloadedStrings #-}

module Invar.Score (
    Plan,
    Target (..),
    Call,
    Permit,
    Report,
    Error (..),
    LogRatio (..),
    prepare,
    targetOf,
    withProbe,
    bind,
    input,
    sourceInspection,
    sourceDigest,
    target,
    binding,
    authorize,
    permission,
    observe,
    admit,
    completion,
    probabilityWords,
    logRatio,
    fullVocabulary,
    fact,
    loadFact,
    describe,
) where

import Control.Monad (unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Object, Value (..), encode, object, toJSON, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8)
import Data.Word (Word32)
import Invar.Artifact qualified as Artifact
import Invar.Infer qualified as Infer
import Invar.Infer.Observation qualified as Inference
import Invar.Infer.Result qualified as Result
import Invar.Infer.Trajectory qualified as Trajectory
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Load qualified as Load
import Invar.Policy qualified as Policy
import Invar.Score.Output (LogRatio (..))
import Invar.Score.Output qualified as Output
import Invar.Score.Probe qualified as Probe
import Invar.Score.Program qualified as Program
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as L
import Invar.Spec.Score qualified as S
import Invar.Tokenizer qualified as Tokenizer
import Numeric.Natural (Natural)

data Plan = Plan Inference.Source Infer.Plan ByteString [Natural]

data Target = Target {targetIdentity :: (String, String, String, String), targetPolicy :: Maybe Policy.Description}
data Call = Call Plan V.Binding ByteString V.Runtime Load.Plan
data Permit = Permit Call ByteString [(ByteString, Object)] V.Runtime L.Fact Policy.Description Tokenizer.Consumption
data Report = Report V.Completion L.Fact Output.Body S.Fact Tokenizer.Consumption
    deriving (Eq, Show)

data Error = Incompatible String | Construction String | Protocol String | Lifecycle V.Error | Loading Load.Error | Registry L.Error | ProcessFailed Int
    deriving (Eq, Show)

targetOf :: Infer.Plan -> Target
targetOf chosen = Target (Infer.artifact requested, Infer.tokenizer requested, Infer.base requested, Infer.assembly requested) (Infer.boundPolicy chosen)
  where
    requested = Infer.requested chosen

prepare :: Inference.Source -> Target -> Either Error Plan
prepare source (Target (artifact, tokenizer, base, assembly) policy) = do
    let before = Trajectory.request (Inference.sourceTrajectory source)
    unbound <- first (Construction . show) (Infer.prepare before {Infer.artifact = artifact, Infer.tokenizer = tokenizer, Infer.base = base, Infer.assembly = assembly})
    planned <- maybe (Right unbound) (\description -> first (Construction . show) (Infer.bindPolicy description unbound)) policy
    pure (Plan source planned (Lazy.toStrict (encode (Inference.describeSource source)) <> "\n") [])

withProbe :: [Natural] -> Plan -> Either Error Plan
withProbe steps (Plan source planned inspection _) = do
    let original = Trajectory.result (Inference.sourceTrajectory source)
        horizon = fromIntegral (length (Result.tokens original)) - Result.promptLength original
    first Incompatible (Probe.selection horizon steps)
    pure (Plan source planned inspection steps)

sourceInspection :: Plan -> ByteString
sourceInspection (Plan _ _ bytes _) = bytes

sourceDigest :: Plan -> String
sourceDigest = Artifact.hex . SHA256.hash . sourceInspection

target :: Plan -> Infer.Plan
target (Plan _ planned _ _) = planned

bind :: V.Binding -> Plan -> Either Error Call
bind bound planned@(Plan source selected _ probes) = do
    let observed = Trajectory.result (Inference.sourceTrajectory source)
        (prefix, response) = splitAt (fromIntegral (Result.promptLength observed)) (Result.tokens observed)
        path = Program.Path (Inference.sourceLog source) (sourceDigest planned) prefix response probes
    (program, runtime) <- first Construction (Program.prepare bound (Infer.requested selected) path)
    loading <- first Loading (Load.prepare bound (Infer.image (Infer.requested selected)))
    pure (Call planned bound program runtime loading)

binding :: Call -> V.Binding
binding (Call _ value _ _ _) = value

inputValue :: Call -> Value
inputValue (Call planned@(Plan _ _ _ probes) bound program _ loading) = Object selected
  where
    fields = Wire.batchFields (bound, program, Load.program loading) (Infer.requested (target planned))
    inspected = Fields.insert "source_inspection" (String (decodeUtf8 (sourceInspection planned))) fields
    selected = if null probes then inspected else Fields.insert "probe_steps" (toJSON probes) inspected

input :: Call -> ByteString
input = Lazy.toStrict . encode . inputValue

authorize :: L.Registry -> Call -> ByteString -> Either Error (L.Registry, Permit)
authorize registry call encoded = records encoded >>= authorizeRecords registry call encoded

authorizeRecords :: L.Registry -> Call -> ByteString -> [(ByteString, Object)] -> Either Error (L.Registry, Permit)
authorizeRecords registry call@(Call planned bound program ready loading) encoded values = do
    let (diagnostics, execution) = span ((`elem` map (Just . String) ["loading", "profile", "load"]) . Fields.lookup "stage" . snd) values
    when (any (Fields.member "phase" . snd) diagnostics) (Left (Protocol "Unexpected phase in score loading diagnostics"))
    (loaded, consumed) <- case execution of
        [(_, loaded), (_, consumed)] -> pure (loaded, consumed)
        _ -> Left (Protocol "Expected exactly one score load and consumption")
    stage "loaded_adapter" loaded
    stage "consumed" consumed
    unless (Object (Fields.delete "stage" consumed) == inputValue call) (Left (Protocol "Score consumption differs from the complete declared call"))
    (description, consumption) <- checkedLoad planned bound loaded
    registered <- first Loading (Load.register loading loaded registry)
    live <- first Registry (L.acquire registered (V.boundInstance bound))
    issued <- first Registry (L.dispatch (L.Dispatch live bound) registered ready)
    intended <- first Lifecycle (V.intent issued (V.boundCall bound))
    current <- first Lifecycle (V.consume (V.Consumption bound program intended) issued)
    loadedFact <- first Registry (L.historical registered (V.boundInstance bound))
    pure (registered, Permit call encoded values current loadedFact description consumption)

permission :: Permit -> ByteString
permission (Permit (Call _ bound program _ _) _ _ _ _ _ _) = Lazy.toStrict (encode (Wire.invocationValue bound program))

observe :: Permit -> ByteString -> Either Error Report
observe permit encoded = do
    suffix <- suffixBytes permit encoded >>= records
    observeRecords permit encoded suffix

suffixBytes :: Permit -> ByteString -> Either Error ByteString
suffixBytes (Permit _ prefix _ _ _ _ _) encoded = do
    unless (prefix `Bytes.isPrefixOf` encoded) (Left (Protocol "Score completion differs from its authorized prefix"))
    pure (Bytes.drop (Bytes.length prefix) encoded)

observeRecords :: Permit -> ByteString -> [(ByteString, Object)] -> Either Error Report
observeRecords (Permit call@(Call (Plan source selected inspection probes) bound _ _ _) _ preceding current loadedFact description consumption) encoded suffix = do
    (raw, result) <- case suffix of
        [(_, before), (_, scoring), (_, after), final] -> do
            mapM_ (uncurry stage) [("verify_before", before), ("cross_score", scoring), ("verify_after", after)]
            pure final
        _ -> Left (Protocol "Expected score verification, execution, verification and one terminal result")
    stage "score_result" result
    parse (Json.fields ["stage", "binding", "observation"]) result
    matching bound result
    value <- parse (.: "observation") result
    let diagnostics = takeWhile ((/= Just (String "loaded_adapter")) . Fields.lookup "stage" . snd) preceding
        measurements = map snd (diagnostics ++ take 3 suffix)
    body <- first Protocol (Output.observe (Output.Input source (Infer.requested selected) inspection description (Tokenizer.consumedCoordinates consumption) probes measurements) value)
    finished <- first Lifecycle (V.finish (binding call) raw current)
    completed <- first Lifecycle (V.completion finished (V.boundAttempt bound))
    case completed of
        Just resultCompletion -> do
            sourceDescription <- first Protocol (Inference.sourcePolicy source)
            let digest = Artifact.hex (SHA256.hash encoded)
                admitted = Inference.sourceTrajectory source
                original = S.Source (Trajectory.request admitted) (Trajectory.binding admitted) sourceDescription (Inference.sourceLog source)
                scored = S.Source (Infer.requested selected) bound description digest
                measured = S.Fact original scored (Output.probabilityWords body) (Output.logRatio body) (Lazy.toStrict (encode (Output.observation body)))
            pure (Report resultCompletion loadedFact body measured consumption)
        Nothing -> Left (Protocol "Score invocation did not complete")

admit :: Call -> Int -> ByteString -> Either Error Report
admit call status encoded = do
    unless (status == 0) (Left (ProcessFailed status))
    values <- records encoded
    let (before, consumedAndRest) = break ((== Just (String "consumed")) . Fields.lookup "stage" . snd) values
    (consumed, suffix) <- case consumedAndRest of
        item : remaining -> pure (item, remaining)
        [] -> Left (Protocol "Missing score consumption")
    let prefixRecords = before ++ [consumed]
    (_, permit) <- authorizeRecords L.empty call (Bytes.unlines (map fst prefixRecords)) prefixRecords
    _ <- suffixBytes permit encoded
    observeRecords permit encoded suffix

completion :: Report -> V.Completion
completion (Report value _ _ _ _) = value

probabilityWords :: Report -> [Word32]
probabilityWords (Report _ _ body _ _) = Output.probabilityWords body

logRatio :: Report -> LogRatio
logRatio (Report _ _ body _ _) = Output.logRatio body

fullVocabulary :: Report -> Maybe S.Distribution
fullVocabulary (Report _ _ body _ _) = Output.fullVocabulary body

loadFact :: Report -> L.Fact
loadFact (Report _ value _ _ _) = value

fact :: Report -> S.Fact
fact (Report _ _ _ value _) = value

describe :: Report -> Value
describe (Report completed _ body measured consumption) =
    object
        [ "format" .= (case Output.fullVocabulary body of Nothing -> "invar-bound-path-score-observation-v1"; Just _ -> "invar-bound-distribution-probe-observation-v1" :: String)
        , "log_sha256" .= S.sourceLog (S.target measured)
        , "binding" .= Wire.bindingValue (V.completedBinding completed)
        , "observation" .= Output.observation body
        , "source_minus_target_log_ratio" .= Output.ratioValue (Output.logRatio body)
        , "strength" .= (case Output.fullVocabulary body of Nothing -> "finite_path_observation"; Just _ -> "finite_full_vocabulary_observation" :: String)
        , "use_admission" .= ("not_evaluated" :: String)
        , "consumption" .= Tokenizer.consumptionValue consumption
        , "unestablished" .= (["execution-report authenticity", "source behavior measurement", "target behavior measurement", "own-cache execution and lineage", "sampler coupling", "full-vocabulary KL", "population guarantee", "use admission"] :: [String])
        ]

checkedLoad :: Plan -> V.Binding -> Object -> Either Error (Policy.Description, Tokenizer.Consumption)
checkedLoad planned@(Plan origin _ _ _) bound fields = do
    parse (Json.fields ["stage", "binding", "load", "image", "requested", "consumed", "tokenizer", "tokenizer_description", "coordinates", "base", "assembly", "model", "revision"]) fields
    matching bound fields
    let requested = Infer.requested (target planned)
    mapM_
        ( \(key, expected) ->
            parse
                ( \value -> do
                    actual <- value .: key >>= Json.identity
                    unless (actual == expected) (fail "Score load differs from the requested materialization")
                )
                fields
        )
        [("requested", Infer.artifact requested), ("consumed", Infer.artifact requested), ("tokenizer", Infer.tokenizer requested), ("base", Infer.base requested), ("assembly", Infer.assembly requested)]
    source <- parse (\value -> (,) <$> value .: "model" <*> value .: "revision") fields
    description <- first Protocol (Policy.describe source (Infer.artifact requested, Infer.tokenizer requested, Infer.base requested, Infer.assembly requested))
    case Infer.boundPolicy (target planned) of
        Just expected -> unless (description == expected) (Left (Protocol "Score model description differs from its bound policy"))
        Nothing -> pure ()
    described <- parse (\value -> value .: "tokenizer_description" >>= either fail pure . Tokenizer.describe) fields
    unless (Tokenizer.identity described == Infer.tokenizer requested) (Left (Protocol "Score tokenizer description differs from the requested target tokenizer"))
    bounds <- parse (\value -> value .: "coordinates" >>= either fail pure . Tokenizer.coordinates) fields
    let admitted = Inference.sourceTrajectory origin
    consumption <- first Incompatible (Tokenizer.consume (Trajectory.digest admitted) (Trajectory.tokenSpace admitted) described bounds (Trajectory.promptTokens admitted, Trajectory.responseTokens admitted))
    pure (description, consumption)

records :: ByteString -> Either Error [(ByteString, Object)]
records encoded = do
    unless ("\n" `Bytes.isSuffixOf` encoded) (Left (Protocol "Incomplete final score observation line"))
    traverse (\bytes -> (,) bytes <$> first Protocol (Json.decode bytes >>= parseEither (withObject "score event" pure))) (Bytes.lines encoded)

stage :: Text -> Object -> Either Error ()
stage expected fields = do
    actual <- parse (.: "stage") fields
    unless (actual == expected && not (Fields.member "phase" fields)) (Left (Protocol "Missing or reordered score observation stage"))

matching :: V.Binding -> Object -> Either Error ()
matching bound fields = do
    actual <- parse Wire.binding fields
    unless (actual == bound) (Left (Lifecycle (V.BindingMismatch bound actual)))

parse :: (Object -> Parser value) -> Object -> Either Error value
parse parser = first Protocol . parseEither parser
