{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Protocol (Result, Permit, Error (..), observe, authorize, authorizeResident, validateSummary, loadProgram, loadedFact, completion, request, adapter, learner, gradients, probabilities) where

import Control.Monad (foldM, unless, void)
import Data.Aeson (Object, Value (..), eitherDecodeStrict, object, withObject, (.:), (.=))
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text.Encoding (encodeUtf8)
import Invar.Infer.Wire qualified as Binding
import Invar.Learn.Wire qualified as Wire
import Invar.Load qualified as Load
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as L
import Numeric.Natural (Natural)

data Artifacts = Artifacts String String (String, String)
    deriving (Eq, Show)

data Result = Result V.Completion Value Artifacts
    deriving (Eq, Show)

data Error = Malformed String | Unexpected String | Mismatch String | Lowering Wire.Error | Lifecycle V.Error | Loading Load.Error | Registry L.Error
    deriving (Eq, Show)

data Context = Context {bound :: V.Binding, numerical :: Value, loading :: Load.Plan, resident :: Bool}
data Progress = Awaiting V.Runtime L.Registry | Loaded V.Runtime L.Registry L.Fact | Consumed V.Runtime L.Registry L.Fact Value | Finished Result

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
        Consumed _ updated fact _ -> Right (updated, Permit context output current fact)
        _ -> Left (Unexpected "Update input has not been consumed")

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
    pure (Consumed accepted registry fact actual)
consumed _ _ _ = Left (Unexpected "Duplicate consumption or consumption before learner load")

finished :: Context -> (Progress, Object) -> ByteString -> Either Error Progress
finished context (Consumed runtime _ _ actual, value) encoded = do
    binding <- matching (bound context) value
    _ <- matchingRequest actual value
    artifacts <- summary actual value
    final <- lifecycle (V.finish binding encoded runtime)
    reported <- lifecycle (V.completion final (V.boundAttempt binding))
    case reported of
        Just completed -> Right (Finished (Result completed actual artifacts))
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

-- This predicate shares result validation without constructing lifecycle facts.
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
identity value = length value == digestLength && all (`elem` (['0' .. '9'] ++ ['a' .. 'f'])) value
  where
    digestLength = 64

diagnostic :: String -> Progress -> Either Error Progress
diagnostic stage progress@(Awaiting _ _) | stage `elem` ["loading", "profile", "load"] = Right progress
diagnostic stage progress@Loaded {} | stage `elem` ["probability_roles", "roles"] = Right progress
diagnostic "reward_update" progress@Consumed {} = Right progress
diagnostic stage _ = Left (Unexpected ("Unknown or misplaced worker stage: " ++ stage))

matching :: V.Binding -> Object -> Either Error V.Binding
matching expected value = do
    actual <- parse Binding.binding value
    unless (actual == expected) (Left (Lifecycle (V.BindingMismatch expected actual)))
    pure actual

matchingRequest :: Value -> Object -> Either Error Value
matchingRequest expected value = do
    actual <- parse (.: "request") value
    unless (actual == expected) (Left (Mismatch "Actual update input differs from the checked lowering"))
    pure actual

parse :: (input -> Parser value) -> input -> Either Error value
parse parser = either (Left . Malformed) Right . parseEither parser

lifecycle :: Either V.Error value -> Either Error value
lifecycle = either (Left . Lifecycle) Right

registryError :: Either L.Error value -> Either Error value
registryError = either (Left . Registry) Right

completion :: Result -> V.Completion
completion (Result result _ _) = result

request :: Result -> Value
request (Result _ actual _) = actual

adapter :: Result -> String
adapter (Result _ _ (Artifacts policy _ _)) = policy

learner :: Result -> String
learner (Result _ _ (Artifacts _ checkpoint _)) = checkpoint

gradients :: Result -> String
gradients (Result _ _ (Artifacts _ _ (observation, _))) = observation

probabilities :: Result -> String
probabilities (Result _ _ (Artifacts _ _ (_, observation))) = observation
