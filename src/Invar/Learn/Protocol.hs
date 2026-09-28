{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Protocol (Result, Permit, Error (..), observe, authorize, authorizeResident, respond, replay, validateSummary, loadProgram, loadedFact, completion, request, checkedRequest, stream, adapter, learner, gradients, probabilities) where

import Control.Monad (foldM, unless, void)
import Data.Aeson (Object, Value (..), eitherDecodeStrict, encode, object, withObject, (.:), (.=))
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Text.Encoding (encodeUtf8)
import Invar.Digest qualified as Digest
import Invar.Infer.Wire qualified as Binding
import Invar.Learn.Objective qualified as Objective
import Invar.Learn.Request qualified as Request
import Invar.Learn.Stream qualified as S
import Invar.Learn.Wire qualified as Wire
import Invar.Load qualified as Load
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as L
import Numeric.Natural (Natural)

data Artifacts = Artifacts String String (String, String)
    deriving (Eq, Show)

data Result = Result V.Completion Request.Request Artifacts S.Stream
    deriving (Eq, Show)

data Error = Malformed String | Unexpected String | Mismatch String | Lowering Wire.Error | Lifecycle V.Error | Loading Load.Error | Registry L.Error | Step S.Error
    deriving (Eq, Show)

data Context = Context {bound :: V.Binding, numerical :: Value, loading :: Load.Plan, resident :: Bool}
data Progress = Awaiting V.Runtime L.Registry | Loaded V.Runtime L.Registry L.Fact | Consumed V.Runtime L.Registry L.Fact Request.Request S.Stream (Maybe S.Reply) | Finished Result

data Permit = Permit Context ByteString Progress L.Fact

observe :: Permit -> ByteString -> Either Error Result
observe (Permit context prefix accepted _) output = do
    unless (prefix `Bytes.isPrefixOf` output) (Left (Mismatch "Completed stream differs from the authorized prefix"))
    final <- foldM (advance context) accepted (Bytes.lines (Bytes.drop (Bytes.length prefix) output))
    case final of
        Finished result -> Right result
        _ -> Left (Unexpected "Worker output ended without a complete bound update")

authorize :: L.Registry -> (V.Binding, V.Runtime) -> ByteString -> Either Error (L.Registry, Permit)
authorize = authorizeWith False

authorizeResident :: L.Registry -> (V.Binding, V.Runtime) -> ByteString -> Either Error (L.Registry, Permit)
authorizeResident = authorizeWith True

authorizeWith :: Bool -> L.Registry -> (V.Binding, V.Runtime) -> ByteString -> Either Error (L.Registry, Permit)
authorizeWith residency registry selection output = do
    (context, current) <- scan residency registry selection output
    case current of
        Consumed _ updated fact _ _ _ -> Right (updated, Permit context output current fact)
        _ -> Left (Unexpected "Update input has not been consumed")

respond :: Permit -> ByteString -> Either Error (Permit, ByteString)
respond (Permit context prefix accepted fact) output = do
    unless (prefix `Bytes.isPrefixOf` output) (Left (Mismatch "Reported step differs from the authorized prefix"))
    advanced <- foldM (advance context) accepted (Bytes.lines (Bytes.drop (Bytes.length prefix) output))
    case advanced of
        Consumed _ _ _ _ _ (Just reply) -> Right (Permit context output advanced fact, Lazy.toStrict (encode (cotangents (bound context) reply)))
        _ -> Left (Unexpected "A reply was requested for a record that does not report a learner step")

replay :: Request.Request -> [Object] -> Either Error String
replay actual records = do
    begun <- declaredSteps actual
    finished' <- foldM follow begun records
    step' (S.complete finished')
  where
    follow current value = do
        name <- parse (.: "stage") value
        fst <$> record name current value

cotangents :: V.Binding -> S.Reply -> Value
cotangents binding reply = object ["stage" .= ("cotangents" :: String), "binding" .= Binding.bindingValue binding, "step" .= S.replyStep reply, "sample" .= S.replySample reply, "observation" .= S.replyObservation reply, "state" .= S.replyState reply, "objective" .= S.objective reply, "reward" .= S.reward reply]

loadedFact :: Permit -> L.Fact
loadedFact (Permit _ _ _ fact) = fact

scan :: Bool -> L.Registry -> (V.Binding, V.Runtime) -> ByteString -> Either Error (Context, Progress)
scan residency registry selection@(_, runtime) output = do
    prepared <- prepare selection
    let context = prepared {resident = residency}
    current <- foldM (advance context) (Awaiting runtime registry) (Bytes.lines output)
    pure (context, current)

loadProgram :: (V.Binding, V.Runtime) -> Either Error ByteString
loadProgram selection = Load.program . loading <$> prepare selection

prepare :: (V.Binding, V.Runtime) -> Either Error Context
prepare (binding, runtime) = do
    command <- lifecycle (V.intent runtime (V.boundCall binding))
    expected <- either (Left . Lowering) Right (Wire.lower command)
    image <- either (Left . Lowering) Right (Wire.image command)
    planned <- either (Left . Loading) Right (Load.prepare binding image)
    pure (Context binding expected planned False)

advance :: Context -> Progress -> ByteString -> Either Error Progress
advance _ (Finished _) _ = Left (Unexpected "Output follows the completed update")
advance context progress encoded = do
    value <- either (Left . Malformed) Right (eitherDecodeStrict encoded)
    stage <- parse (.: "stage") value
    case stage :: String of
        "loaded_learner" -> loaded context progress value
        "consumed" -> consumed context progress value
        "result" -> finished context (progress, value) encoded
        _ | stage `elem` ["proximal", "current", "applied"] -> stepping context progress stage value
        "activation" | resident context, Awaiting {} <- progress -> Right progress
        _ -> diagnostic stage progress

loaded :: Context -> Progress -> Object -> Either Error Progress
loaded context (Awaiting runtime registry) value = do
    _ <- matching (bound context) value
    reported <- parse (.: "state") value
    intended <- parse (withObject "update input" state) (numerical context)
    unless (reported == intended) (Left (Mismatch "Loaded learner tokenizer reference or optimizer differs from the update input"))
    updated <- either (Left . Loading) Right (Load.register (loading context) value registry)
    live <- registryError (L.acquire updated (V.boundInstance (bound context)))
    issued <- registryError (L.dispatch (L.Dispatch live (bound context)) updated runtime)
    fact <- registryError (L.historical updated (V.boundInstance (bound context)))
    pure (Loaded issued updated fact)
  where
    state entry = do
        policy <- entry .: "policy" :: Parser Value
        checkpoint <- entry .: "learner" :: Parser Value
        tokenizer <- entry .: "tokenizer" :: Parser Value
        base <- entry .: "base" :: Parser Value
        assembly <- entry .: "assembly" :: Parser Value
        reference <- entry .: "reference" :: Parser Value
        optimizer <- entry .: "optimizer" :: Parser Value
        pure (object ["policy" .= policy, "learner" .= checkpoint, "tokenizer" .= tokenizer, "base" .= base, "assembly" .= assembly, "reference" .= reference, "optimizer" .= optimizer])
loaded _ _ _ = Left (Unexpected "Duplicate or misplaced learner load report")

consumed :: Context -> Progress -> Object -> Either Error Progress
consumed context (Loaded runtime registry fact) value = do
    actualLoad <- parse (\fields -> fields .: "load" >>= withObject "consumed learner load" Load.invocation) value
    unless (actualLoad == (bound context, Load.program (loading context))) (Left (Mismatch "Update consumption names a different learner load"))
    binding <- matching (bound context) value
    program <- encodeUtf8 <$> parse (.: "program") value
    actual <- matchingRequest (numerical context) value
    command <- lifecycle (V.intent runtime (V.boundCall binding))
    accepted <- lifecycle (V.consume (V.Consumption binding program command) runtime)
    begun <- declaredSteps actual
    pure (Consumed accepted registry fact actual begun Nothing)
consumed _ _ _ = Left (Unexpected "Duplicate consumption or consumption before learner load")

finished :: Context -> (Progress, Object) -> ByteString -> Either Error Progress
finished context (Consumed runtime _ _ actual steps _, value) encoded = do
    binding <- matching (bound context) value
    _ <- matchingRequest (Request.value actual) value
    artifacts@(Artifacts policy _ _) <- summary (Request.value actual) value
    ended <- step' (S.complete steps)
    unless (ended == policy) (Left (Mismatch "The last applied step does not end at the staged adapter"))
    final <- lifecycle (V.finish binding encoded runtime)
    reported <- lifecycle (V.completion final (V.boundAttempt binding))
    case reported of
        Just completed -> Right (Finished (Result completed actual artifacts steps))
        Nothing -> Left (Unexpected "Bound update did not produce a completion")
finished _ _ _ = Left (Unexpected "Update result arrived without consumption")

summary :: Value -> Object -> Either Error Artifacts
summary actual value = do
    policy <- parse (.: "adapter") value
    checkpoint <- parse (.: "learner") value
    observation <- parse (.: "gradients") value
    unless (identity observation) (Left (Mismatch "Invalid gradient observation identity"))
    probability <- parse (.: "probabilities") value
    unless (identity probability) (Left (Mismatch "Invalid probability observation identity"))
    storage <- parse (.: "storage") value
    unless (identity policy && identity checkpoint && storage == ("staged; not published" :: String)) (Left (Mismatch "Invalid staged checkpoint identity or storage status"))
    update <- parse (.: "update") value
    previous <- parse (withObject "update input" (\entry -> entry .: "policy" :: Parser String)) actual
    before <- parse (.: "before") update
    after <- parse (.: "after") update
    unless (before == previous && after == policy) (Left (Mismatch "Update and materialized adapter identities disagree"))
    validateStats actual update
    pure (Artifacts policy checkpoint (observation, probability))

validateSummary :: Value -> Object -> Either Error ()
validateSummary actual value = void (summary actual value)

validateStats :: Value -> Object -> Either Error ()
validateStats actual value = do
    samples <- parse (withObject "update input" (\entry -> entry .: "samples" :: Parser [Object])) actual
    counts <- traverse (parse (\sample -> length <$> (sample .: "behavior_bits" :: Parser [Natural]))) samples
    tokens <- parse (.: "active_tokens") value
    nonzero <- parse (.: "nonzero_advantages") value
    loss <- parse (.: "loss") value
    gradient <- parse (.: "gradient_norm") value
    rewardGradient <- parse (.: "reward_gradient_norm") value
    unless (tokens == sum counts && tokens > 0 && nonzero >= (0 :: Int) && nonzero <= length samples) (Left (Mismatch "Update sample or active-token counts disagree"))
    unless (all finite [loss, gradient, rewardGradient] && gradient >= 0 && rewardGradient >= 0) (Left (Mismatch "Invalid numerical update summary"))
  where
    finite number = not (isNaN (number :: Double) || isInfinite number)

identity :: String -> Bool
identity = Digest.sha256

diagnostic :: String -> Progress -> Either Error Progress
diagnostic stage progress@(Awaiting _ _) | stage `elem` ["loading", "profile", "load"] = Right progress
diagnostic stage (Consumed runtime registry fact actual steps _) | stage `elem` ["reward_update", "artifacts", "checkpoint"] = Right (Consumed runtime registry fact actual steps Nothing)
diagnostic stage _ = Left (Unexpected ("Unknown or misplaced worker stage: " ++ stage))

matching :: V.Binding -> Object -> Either Error V.Binding
matching expected value = do
    actual <- parse Binding.binding value
    unless (actual == expected) (Left (Lifecycle (V.BindingMismatch expected actual)))
    pure actual

matchingRequest :: Value -> Object -> Either Error Request.Request
matchingRequest expected value = do
    actual <- parse (.: "request") value
    unless (actual == expected) (Left (Mismatch "Actual update input differs from the checked lowering"))
    parse Request.parse actual

parse :: (input -> Parser value) -> input -> Either Error value
parse parser = either (Left . Malformed) Right . parseEither parser

lifecycle :: Either V.Error value -> Either Error value
lifecycle = either (Left . Lifecycle) Right

registryError :: Either L.Error value -> Either Error value
registryError = either (Left . Registry) Right

completion :: Result -> V.Completion
completion (Result result _ _ _) = result

request :: Result -> Value
request (Result _ actual _ _) = Request.value actual

checkedRequest :: Result -> Request.Request
checkedRequest (Result _ actual _ _) = actual

stream :: Result -> S.Stream
stream (Result _ _ _ steps) = steps

adapter :: Result -> String
adapter (Result _ _ (Artifacts policy _ _) _) = policy

learner :: Result -> String
learner (Result _ _ (Artifacts _ checkpoint _) _) = checkpoint

gradients :: Result -> String
gradients (Result _ _ (Artifacts _ _ (observation, _)) _) = observation

probabilities :: Result -> String
probabilities (Result _ _ (Artifacts _ _ (_, observation)) _) = observation

declaredSteps :: Request.Request -> Either Error S.Stream
declaredSteps actual = parse (withObject "checked update request" plan) (Request.value actual)
  where
    plan fields = do
        entries <- fields .: "samples"
        declared <- traverse entry entries
        steps <- fields .: "steps"
        policy <- fields .: "policy"
        profile <- Objective.Profile <$> fields .: "epsilon" <*> fields .: "penalty"
        pure (S.begin profile policy declared steps)
    entry = withObject "update sample" $ \fields -> S.Sample <$> fields .: "sample" <*> fields .: "behavior_bits" <*> fields .: "reference_bits" <*> fields .: "advantage_bits"

stepping :: Context -> Progress -> String -> Object -> Either Error Progress
stepping context (Consumed runtime registry fact actual steps _) stage value = do
    _ <- matching (bound context) value
    (updated, reply) <- record stage steps value
    pure (Consumed runtime registry fact actual updated reply)
stepping _ _ _ _ = Left (Unexpected "A learner step was reported before consumption")

record :: String -> S.Stream -> Object -> Either Error (S.Stream, Maybe S.Reply)
record "proximal" steps value = (,Nothing) <$> (step' =<< S.proximal steps <$> parse (.: "sample") value <*> parse (.: "words") value)
record "current" steps value = do
    report <- parse (\fields -> S.Current <$> fields .: "step" <*> fields .: "sample" <*> fields .: "words" <*> fields .: "observation" <*> fields .: "state") value
    fmap Just <$> step' (S.current steps report)
record "applied" steps value = do
    report <- S.Applied <$> parse (.: "step") value <*> parse (.: "before") value <*> parse (.: "after") value <*> parse (.: "consumed") value
    (,Nothing) <$> step' (S.applied steps report)
record name _ _ = Left (Unexpected ("Unexpected learner step record: " ++ name))

step' :: Either S.Error value -> Either Error value
step' = either (Left . Step) Right
