{-# LANGUAGE OverloadedStrings #-}

module Probabilities (probabilities, fixture, observed, decoded) where

import Control.Monad (forM_)
import Data.Aeson (FromJSON, Result (..), Value (..), encode, fromJSON, object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Foldable (toList)
import Data.List (uncons)
import Data.Map.Strict qualified as Map
import Data.Text qualified as Text
import Data.Word (Word32)
import GHC.Float (castFloatToWord32)
import Hedgehog
import Invar.Artifact qualified as Artifact
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
probabilities = Group "Bound learner observations" [("probability artifact identity is mandatory", once required), ("complete file binds every learner observation to the consumed input", once valid), ("matching hashes cannot hide invalid observations", once malformed), ("duplicate JSON keys and trailing bytes are rejected", once ambiguous), ("changed missing and symbolic files are rejected", once files), ("proximal is the first observation and the file repeats the reported steps", once engine), ("a later step current must repeat the reported step", once laterStep)]
  where
    once = withTests 1 . property

fixture :: [Value] -> Value
fixture = fixtureWith (\_ _ word -> word)

fixtureWith :: (Text.Text -> Text.Text -> Word32 -> Word32) -> [Value] -> Value
fixtureWith role events = object ["format" .= String "invar-probabilities-v5", "invocation" .= invocation, "request" .= request, "samples" .= map sample ordered]
  where
    consumed = events !! 1
    invocation = object ["binding" .= field "binding" consumed, "program" .= field "program" consumed]
    request = field "request" consumed
    delivered = array (field "samples" request)
    ordered = [item | name <- array (field "order" request), item <- delivered, field "sample" item == name]
    plan = decoded (field "steps" request) :: [[Text.Text]]
    named item = decoded (field "sample" item) :: Text.Text
    behaviors item = decoded (field "behavior_bits" item) :: [Word32]
    current item = map (role "current" (named item)) (behaviors item)
    proximal item = map (role "proximal" (named item)) (behaviors item)
    sample item = object ["sample" .= field "sample" item, "dtype" .= String "F32", "proximal" .= proximal item, "steps" .= [object ["step" .= position, "current" .= current item] | (position, batch) <- zip [0 :: Int ..] plan, named item `elem` batch]]

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

malformed :: PropertyT IO ()
malformed = do
    configured@(_, events) <- matched
    let original = fixture events
    (firstSample, remaining) <- evalMaybe (uncons (array (field "samples" original)))
    (firstStep, laterSteps) <- evalMaybe (uncons (array (field "steps" firstSample)))
    let replaceSample sample = change "samples" (toJSON (sample : remaining)) original
        replaceStep value = replaceSample (change "steps" (toJSON (value : laterSteps)) firstSample)
        wrongFields = [("dtype", String "F64"), ("sample", String "unknown"), ("behavior", toJSON [Number 0]), ("proximal", toJSON ([] :: [Value])), ("proximal", toJSON [Number nanWord]), ("proximal", toJSON [Number positiveWord]), ("proximal", toJSON [Number wordLimit]), ("steps", toJSON ([] :: [Value]))]
        changes = [change "invocation" Null original, change "request" Null original, change "format" Null original, change "format" (String "invar-probabilities-v3") original, change "samples" (toJSON ([] :: [Value])) original, change "extra" Null original] ++ [replaceSample (change name value firstSample) | (name, value) <- wrongFields] ++ [replaceSample (change "extra" Null firstSample), replaceStep (change "current" (toJSON [Number 0]) firstStep), replaceStep (change "step" (Number 1) firstStep), replaceStep (change "extra" Null firstStep)]
    forM_ changes $ \value -> reject configured (Lazy.toStrict (encode value))
  where
    wordLimit = 4294967296
    nanWord = 2143289344
    positiveWord = 1065353216

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
    reject configured ("{\"format\":\"invar-probabilities-v5\"," <> Bytes.drop 1 encoded)
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
    configured@(_, events) <- setup
    let other = castFloatToWord32 (-3)
        shifted target name _ word = if name == target then other else word
    root <- workspace
    result <- observed root configured (Lazy.toStrict (encode (fixture events)))
    evalIO (Worker.verifyProbabilities root result) >>= (=== Right ())
    forM_ [fixtureWith (shifted "proximal") events, fixtureWith (shifted "current") events] $ \value ->
        reject configured (Lazy.toStrict (encode value))

laterStep :: PropertyT IO ()
laterStep = do
    let algorithm = Value.Record (Map.fromList [("epsilon", Value.Atom (Value.Number (1 / 5))), ("penalty", Value.Atom (Value.Number (1 / 25))), ("delta", Value.Atom (Value.Number (1 / 10000))), ("steps", Value.Atom (Value.Number 2))])
    configured@(_, events) <- setupFor (Map.insert (Source.Semantic "algorithm") algorithm sameReference)
    let other = castFloatToWord32 (-3)
        later = fixtureWith (\name sample word -> if name == "current" && sample == "s1" then other else word) events
    root <- workspace
    accepted <- observed root configured (Lazy.toStrict (encode (fixture events)))
    evalIO (Worker.verifyProbabilities root accepted) >>= (=== Right ())
    reject configured (Lazy.toStrict (encode later))
