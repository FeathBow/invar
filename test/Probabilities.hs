{-# LANGUAGE OverloadedStrings #-}

module Probabilities (probabilities, fixture, observed) where

import Control.Monad (forM_)
import Data.Aeson (Value (..), encode, object, toJSON, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Foldable (toList)
import Data.List (uncons)
import Data.Text qualified as Text
import Hedgehog
import Invar.Artifact qualified as Artifact
import Invar.Learn.Protocol qualified as P
import Invar.Learn.Worker qualified as Worker
import Invar.Spec.Invocation qualified as V
import Store (workspace)
import System.FilePath ((</>))
import System.IO.Error (isDoesNotExistError, tryIOError)
import System.Posix.Files (createSymbolicLink)
import Updates (alter, change, field, observe, setup, wire)

type UpdatesContext = (V.Binding, V.Runtime)

probabilities :: Group
probabilities = Group "Bound objective inputs" [("probability artifact identity is mandatory", once required), ("complete file binds every objective role and consumed input", once valid), ("matching hashes cannot hide invalid observations", once malformed), ("duplicate JSON keys and trailing bytes are rejected", once ambiguous), ("changed missing and symbolic files are rejected", once files)]
  where
    once = withTests 1 . property

fixture :: [Value] -> Value
fixture events = object ["format" .= String "invar-probabilities-v1", "invocation" .= invocation, "request" .= request, "samples" .= samples]
  where
    consumed = events !! 1
    invocation = object ["binding" .= field "binding" consumed, "program" .= field "program" consumed]
    request = field "request" consumed
    delivered = array (field "samples" request)
    samples = [sample item | name <- array (field "order" request), item <- delivered, field "sample" item == name]
    sample item = object (["sample" .= field "sample" item, "dtype" .= String "F32", "active" .= map (const True) (array (field "behavior_bits" item))] ++ [name .= field "behavior_bits" item | name <- ["behavior", "proximal", "reference", "current", "advantage"]])

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
    configured@(_, events) <- setup
    root <- workspace
    result <- observed root configured (Lazy.toStrict (encode (fixture events)))
    evalIO (Worker.verifyProbabilities root result) >>= (=== Right ())

malformed :: PropertyT IO ()
malformed = do
    configured@(_, events) <- setup
    (firstSample, remaining) <- evalMaybe (uncons (array (field "samples" (fixture events))))
    let original = fixture events
        replaceSample sample = change "samples" (toJSON (sample : remaining)) original
        wrongFields = [("dtype", String "F64"), ("sample", String "unknown"), ("behavior", toJSON [Number 0]), ("active", toJSON [Bool False]), ("active", toJSON [Number 1]), ("current", toJSON ([] :: [Value])), ("proximal", toJSON [Number (-1)]), ("reference", toJSON [Number wordLimit]), ("advantage", toJSON [Number nanWord]), ("current", toJSON [Number positiveWord]), ("current", toJSON [Bool False])]
        changes = [change "invocation" Null original, change "request" Null original, change "format" Null original, change "samples" (toJSON ([] :: [Value])) original, change "extra" Null original] ++ [replaceSample (change name value firstSample) | (name, value) <- wrongFields]
    forM_ changes $ \value -> reject configured (Lazy.toStrict (encode value))
  where
    wordLimit = 4294967296
    nanWord = 2143289344
    positiveWord = 1065353216

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
    configured@(_, events) <- setup
    let encoded = Lazy.toStrict (encode (fixture events))
    reject configured ("{\"format\":\"invar-probabilities-v1\"," <> Bytes.drop 1 encoded)
    reject configured (encoded <> " null")
    reject configured (replaceBytes "\"dtype\":\"F32\"" "\"dtype\":\"F32\",\"dtype\":\"F32\"" encoded)

replaceBytes :: ByteString -> ByteString -> ByteString -> ByteString
replaceBytes old new encoded = let (prefix, suffix) = Bytes.breakSubstring old encoded in prefix <> new <> Bytes.drop (Bytes.length old) suffix

files :: PropertyT IO ()
files = do
    configured@(_, events) <- setup
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
