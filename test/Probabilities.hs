{-# LANGUAGE OverloadedStrings #-}

module Probabilities (probabilities, fixture, observed, decoded) where

import Control.Monad (forM_)
import Data.Aeson (FromJSON, Result (..), Value (..), encode, fromJSON, object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Bits (xor)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Foldable (toList)
import Data.List (uncons, zip4)
import Data.Map.Strict qualified as Map
import Data.Text qualified as Text
import Data.Word (Word32)
import GHC.Float (castFloatToWord32, castWord32ToFloat, float2Double)
import Hedgehog
import Invar.Artifact qualified as Artifact
import Invar.Learn.Objective qualified as Objective
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Worker qualified as Worker
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Program qualified as Source
import Invar.Spec.Value qualified as Value
import Learning (world)
import Store (workspace)
import System.FilePath ((</>))
import System.IO.Error (isDoesNotExistError, tryIOError)
import System.Posix.Files (createSymbolicLink)
import Updates (alter, change, field, observe, setup, setupFor, wire)

type UpdatesContext = (V.Binding, V.Runtime)

probabilities :: Group
probabilities = Group "Bound objective inputs" [("probability artifact identity is mandatory", once required), ("complete file binds every objective role and consumed input", once valid), ("matching hashes cannot hide invalid observations", once malformed), ("core zero advantage admits only the matching sign bit", once signedZero), ("matching hashes cannot hide scalar terms slopes or mean", once scalars), ("update summary must equal the core scalar loss", once summary), ("raw summary zero sign survives JSON admission", once summarySign), ("duplicate JSON keys and trailing bytes are rejected", once ambiguous), ("changed missing and symbolic files are rejected", once files), ("proximal is the first observation, reference is the engine score and the file repeats the reported steps", once engine), ("a later step current must repeat the reported step", once laterStep)]
  where
    once = withTests 1 . property

fixture :: [Value] -> Value
fixture = fixtureWith (\_ _ word -> word)

fixtureWith :: (Text.Text -> Text.Text -> Word32 -> Word32) -> [Value] -> Value
fixtureWith role events = object ["format" .= String "invar-probabilities-v4", "invocation" .= invocation, "request" .= request, "samples" .= map sample ordered, "scalar_reference" .= Objective.reference, "losses" .= losses]
  where
    consumed = events !! 1
    invocation = object ["binding" .= field "binding" consumed, "program" .= field "program" consumed]
    request = field "request" consumed
    delivered = array (field "samples" request)
    ordered = [item | name <- array (field "order" request), item <- delivered, field "sample" item == name]
    plan = decoded (field "steps" request) :: [[Text.Text]]
    named item = decoded (field "sample" item) :: Text.Text
    byName = Map.fromList [(named item, item) | item <- delivered]
    profile = Objective.Profile (decoded (field "epsilon" request)) (decoded (field "penalty" request))
    behaviors item = decoded (field "behavior_bits" item) :: [Word32]
    scores item = let scored = decoded (field "reference_bits" item) :: [Word32] in if null scored then behaviors item else scored
    current item = map (role "current" (named item)) (behaviors item)
    proximal item = map (role "proximal" (named item)) (behaviors item)
    reference item = map (role "reference" (named item)) (scores item)
    denominator batch = sum [length (behaviors item) | name <- batch, Just item <- [Map.lookup name byName]]
    outputs item batch = either (error . show) id (Objective.calculate profile (denominator batch) [Objective.Inputs {Objective.current = c, Objective.proximal = q, Objective.behavior = b, Objective.fixed = r, Objective.advantage = decoded (field "advantage_bits" item)} | (b, q, r, c) <- zip4 (behaviors item) (proximal item) (reference item) (current item)])
    losses = [either (error . show) id (Objective.mean32 (concat [map Objective.term (outputs item batch) | name <- batch, Just item <- [Map.lookup name byName]])) | batch <- plan]
    sample item = object ["sample" .= field "sample" item, "dtype" .= String "F32", "behavior" .= behaviors item, "reference" .= reference item, "advantage" .= field "advantage_bits" item, "proximal" .= proximal item, "steps" .= [entry position item batch | (position, batch) <- zip [0 :: Int ..] plan, named item `elem` batch]]
    entry position item batch = let actual = outputs item batch in object ["step" .= position, "current" .= current item, "objective" .= object ["terms" .= map Objective.term actual, "current_gradient" .= map Objective.gradient actual, "reward_gradient" .= map Objective.rewardGradient actual]]

decoded :: (FromJSON value) => Value -> value
decoded value = case fromJSON value of
    Success result -> result
    Error problem -> error problem

array :: Value -> [Value]
array (Array values) = toList values
array _ = error "Expected fixture array"

observed :: FilePath -> (UpdatesContext, [Value]) -> ByteString -> PropertyT IO P.Result
observed root (context, events) encoded = do
    let path = root </> "probabilities.json"
    evalIO (Bytes.writeFile path encoded)
    digest <- evalIO (Artifact.identity "Probability observation" path)
    evalEither (observe context (wire (alter 2 (change "probabilities" (String (Text.pack digest))) events)))

required :: PropertyT IO ()
required = do
    (context, events) <- setup
    let omit (Object fields) = Object (Fields.delete "probabilities" fields)
        omit value = value
    forM_ [omit, change "probabilities" (String ""), change "probabilities" (String (Text.replicate digestLength "G"))] $ \modify ->
        case observe context (wire (alter 2 modify events)) of
            Left _ -> success
            Right unexpected -> annotateShow unexpected >> failure
  where
    digestLength = 64

valid :: PropertyT IO ()
valid = do
    configured@(_, events) <- matched
    root <- workspace
    result <- observed root configured (Lazy.toStrict (encode (fixture events)))
    evalIO (Worker.verifyProbabilities root result) >>= (=== Right ())

signedZero :: PropertyT IO ()
signedZero = do
    let zero = Value.Atom (Value.Number 0)
        supplied = Map.insert (Source.Semantic "rewards") (Value.Mapping (Map.fromList [(2, zero), (9, zero)])) sameReference
        negativeZero = 2147483648
    (context, initial) <- setupFor supplied
    let events = alter 2 (\item -> change "update" (change "nonzero_advantages" (Number 0) (field "update" item)) item) initial
        configured = (context, events)
        original = fixture events
    root <- workspace
    result <- observed root configured (Lazy.toStrict (encode original))
    evalIO (Worker.verifyProbabilities root result) >>= (=== Right ())
    (firstSample, remaining) <- evalMaybe (uncons (array (field "samples" original)))
    field "advantage" firstSample === Number 0
    let changed = change "samples" (toJSON (change "advantage" (Number negativeZero) firstSample : remaining)) original
    reject configured (Lazy.toStrict (encode changed))

scalars :: PropertyT IO ()
scalars = do
    configured@(_, events) <- matched
    let original = fixture events
    (firstSample, remaining) <- evalMaybe (uncons (array (field "samples" original)))
    (firstStep, laterSteps) <- evalMaybe (uncons (array (field "steps" firstSample)))
    let actual = field "objective" firstStep
        replaceStep value = change "samples" (toJSON (change "steps" (toJSON (value : laterSteps)) firstSample : remaining)) original
        replaceObjective value = replaceStep (change "objective" value firstStep)
    forM_ ["terms", "current_gradient", "reward_gradient"] $ \name -> do
        let words32 = decoded (field name actual) :: [Word32]
        reject configured (Lazy.toStrict (encode (replaceObjective (change name (toJSON (map (`xor` 1) words32)) actual))))
        reject configured (Lazy.toStrict (encode (replaceObjective (change name (toJSON [Bool False]) actual))))
    let changes = [replaceStep (change "current" (toJSON [Number 0]) firstStep), replaceStep (change "step" (Number 1) firstStep), replaceObjective Null, replaceObjective (change "extra" Null actual), change "scalar_reference" (String "unqualified") original, change "losses" (toJSON [Number 1]) original, change "format" (String "invar-probabilities-v3") original]
    forM_ changes $ \value -> reject configured (Lazy.toStrict (encode value))

summary :: PropertyT IO ()
summary = do
    (context, events) <- matched
    let changed = alter 2 (\item -> change "update" (change "loss" (Number 0.5) (field "update" item)) item) events
    reject (context, changed) (Lazy.toStrict (encode (fixture changed)))

summarySign :: PropertyT IO ()
summarySign = do
    (context, events) <- matched
    root <- workspace
    let path = root </> "probabilities.json"
    evalIO (Bytes.writeFile path (Lazy.toStrict (encode (fixture events))))
    digest <- evalIO (Artifact.identity "Probability observation" path)
    let encoded = wire (alter 2 (change "probabilities" (String (Text.pack digest))) events)
    forM_ ["-0.0", "-0e0", "-1e-999"] $ \literal -> do
        result <- evalEither (observe context (replaceBytes "\"loss\":0" ("\"loss\":" <> literal) encoded))
        actual <- evalIO (Worker.verifyProbabilities root result)
        case actual of
            Left (Worker.InvalidProbability _) -> success
            unexpected -> annotateShow unexpected >> failure
    result <- evalEither (observe context (replaceBytes "\"loss\":0" "\"l\\u006fss\":-0" encoded))
    evalIO (Worker.verifyProbabilities root result) >>= (=== Right ())

malformed :: PropertyT IO ()
malformed = do
    configured@(_, events) <- matched
    (firstSample, remaining) <- evalMaybe (uncons (array (field "samples" (fixture events))))
    let original = fixture events
        replaceSample sample = change "samples" (toJSON (sample : remaining)) original
        wrongFields = [("dtype", String "F64"), ("sample", String "unknown"), ("behavior", toJSON [Number 0]), ("proximal", toJSON ([] :: [Value])), ("proximal", toJSON [Number nanWord]), ("proximal", toJSON [Number positiveWord]), ("proximal", toJSON [Number wordLimit]), ("reference", toJSON [Number negativeZero, Number negativeZero]), ("advantage", Number positiveWord), ("advantage", toJSON [Number 0]), ("steps", toJSON ([] :: [Value]))]
        changes = [change "invocation" Null original, change "request" Null original, change "format" Null original, change "samples" (toJSON ([] :: [Value])) original, change "extra" Null original] ++ [replaceSample (change name value firstSample) | (name, value) <- wrongFields] ++ [replaceSample (change "extra" Null firstSample)]
    forM_ changes $ \value -> reject configured (Lazy.toStrict (encode value))
  where
    wordLimit = 4294967296
    nanWord = 2143289344
    positiveWord = 1065353216
    negativeZero = 2147483648

matched :: PropertyT IO (UpdatesContext, [Value])
matched = setupFor sameReference

sameReference :: E.World
sameReference = Map.insert (Source.Semantic "reference_scores") (world Map.! Source.Semantic "behavior") world

reject :: (UpdatesContext, [Value]) -> ByteString -> PropertyT IO ()
reject configured encoded = do
    root <- workspace
    result <- observed root configured encoded
    returned <- evalIO (Worker.verifyProbabilities root result)
    case returned of
        Left (Worker.InvalidProbability _) -> success
        unexpected -> annotateShow unexpected >> failure

ambiguous :: PropertyT IO ()
ambiguous = do
    configured@(_, events) <- matched
    let encoded = Lazy.toStrict (encode (fixture events))
    reject configured ("{\"format\":\"invar-probabilities-v4\"," <> Bytes.drop 1 encoded)
    reject configured (encoded <> " null")
    reject configured (replaceBytes "\"dtype\":\"F32\"" "\"dtype\":\"F32\",\"dtype\":\"F32\"" encoded)

replaceBytes :: ByteString -> ByteString -> ByteString -> ByteString
replaceBytes old new encoded = let (prefix, suffix) = Bytes.breakSubstring old encoded in prefix <> new <> Bytes.drop (Bytes.length old) suffix

files :: PropertyT IO ()
files = do
    configured@(_, events) <- matched
    root <- workspace
    result <- observed root configured (Lazy.toStrict (encode (fixture events)))
    evalIO (Bytes.appendFile (root </> "probabilities.json") " ")
    returned <- evalIO (Worker.verifyProbabilities root result)
    case returned of
        Left (Worker.ProbabilityMismatch expected actual) -> assert (expected /= actual)
        unexpected -> annotateShow unexpected >> failure
    missing <- workspace
    absent <- evalIO (tryIOError (Worker.verifyProbabilities missing result))
    case absent of
        Left problem -> assert (isDoesNotExistError problem)
        Right unexpected -> annotateShow unexpected >> failure
    evalIO (createSymbolicLink (root </> "probabilities.json") (missing </> "probabilities.json"))
    linked <- evalIO (tryIOError (Worker.verifyProbabilities missing result))
    case linked of
        Left _ -> success
        Right unexpected -> annotateShow unexpected >> failure

engine :: PropertyT IO ()
engine = do
    (context, events) <- setup
    let other = castFloatToWord32 (-3)
        shifted target name _ word = if name == target then other else word
        build role = let value = fixtureWith role events in (value, (context, reported value))
        reported value = alter 2 (\item -> change "update" (change "loss" (Number (realToFrac (float2Double (castWord32ToFloat (first' (decoded (field "losses" value))))))) (field "update" item)) item) events
        first' values = case values of
            value : _ -> value
            [] -> 0
    let (accepted, configured) = build (\_ _ word -> word)
    root <- workspace
    result <- observed root configured (Lazy.toStrict (encode accepted))
    evalIO (Worker.verifyProbabilities root result) >>= (=== Right ())
    forM_ [build (shifted "proximal"), build (shifted "reference"), build (shifted "current")] $ \(value, configured') ->
        reject configured' (Lazy.toStrict (encode value))

laterStep :: PropertyT IO ()
laterStep = do
    let algorithm = Value.Record (Map.fromList [("epsilon", Value.Atom (Value.Number (1 / 5))), ("penalty", Value.Atom (Value.Number (1 / 25))), ("delta", Value.Atom (Value.Number (1 / 10000))), ("steps", Value.Atom (Value.Number 2))])
    (context, events) <- setupFor (Map.insert (Source.Semantic "algorithm") algorithm sameReference)
    let other = castFloatToWord32 (-3)
        later = fixtureWith (\name sample word -> if name == "current" && sample == "s1" then other else word) events
        firstLoss = case decoded (field "losses" later) :: [Word32] of
            value : _ -> value
            [] -> 0
        reported = alter 2 (\item -> change "update" (change "loss" (Number (realToFrac (float2Double (castWord32ToFloat firstLoss)))) (field "update" item)) item) events
    root <- workspace
    accepted <- observed root (context, reported) (Lazy.toStrict (encode (fixtureWith (\_ _ word -> word) events)))
    evalIO (Worker.verifyProbabilities root accepted) >>= (=== Right ())
    reject (context, reported) (Lazy.toStrict (encode later))
