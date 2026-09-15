{-# LANGUAGE OverloadedStrings #-}

module Policies (policies, artifact) where

import Control.Monad (forM_)
import Data.Aeson (Value (..), encode, object, (.=))
import Data.Aeson.Key qualified as Key
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as TextBytes
import Data.ByteString.Lazy qualified as Lazy
import Data.List (isInfixOf, mapAccumL)
import Hedgehog
import Invar.Learn.Worker qualified as Worker
import Invar.Policy qualified as Policy
import Store (workspace)
import System.Directory (createDirectory)
import System.FilePath ((</>))
import System.IO.Error (ioeGetErrorString, tryIOError)
import System.Posix.Files (createNamedPipe, createSymbolicLink)
import Updates (checkpointResult)

type Tensor = (String, [Integer], ByteString)

policies :: Group
policies = Group "Published tensor identities" [("policy identity matches the independent tensor worker", once golden), ("signed zero remains part of policy identity", once signedZero), ("malformed tensor metadata cannot identify a policy", once malformed), ("non-finite policy words are rejected", once nonfinite), ("checkpoint verification binds both reported artifacts", once checkpoint), ("nonregular artifacts never become verified checkpoints", once nonregular)]
  where
    once = withTests 1 . property

expected :: String
expected = "db4cfd69f8498c61e59e9cc58b4643f43a07974a3bb8a326bddc7a5cdc8183cd"

tensors :: [Tensor]
tensors = [("z", [3], Bytes.pack [0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 160, 63]), ("a", [], Bytes.pack [0, 0, 0, 64]), ("empty", [0, 3], Bytes.empty), ("quoted\"\\\n\DEL文😀", [1], Bytes.pack [0, 0, 32, 192])]

artifact :: [Tensor] -> ByteString
artifact values = framed (Lazy.toStrict (encode (object entries))) content
  where
    (_, entries) = mapAccumL entry 0 values
    content = Bytes.concat [bytes | (_, _, bytes) <- values]
    entry offset (name, shape, bytes) =
        let end = offset + fromIntegral (Bytes.length bytes)
         in (end, Key.fromString name .= object ["dtype" .= String "F32", "shape" .= shape, "data_offsets" .= [offset, end :: Integer]])

framed :: ByteString -> ByteString -> ByteString
framed header content = prefix (fromIntegral (Bytes.length header)) <> header <> content

prefix :: Integer -> ByteString
prefix count = Bytes.pack (take prefixBytes (digits count))
  where
    prefixBytes = 8
    base = 256
    digits value = fromIntegral (value `mod` base) : digits (value `div` base)

write :: ByteString -> PropertyT IO FilePath
write content = do
    root <- workspace
    let path = root </> "adapter.safetensors"
    evalIO (Bytes.writeFile path content)
    pure path

golden :: PropertyT IO ()
golden = forM_ [tensors, reverse tensors] $ \ordered -> do
    path <- write (artifact ordered)
    evalIO (Policy.identity path) >>= (=== expected)

signedZero :: PropertyT IO ()
signedZero = do
    first <- write (artifact [("zero", [1], Bytes.pack [0, 0, 0, 0])])
    second <- write (artifact [("zero", [1], Bytes.pack [0, 0, 0, 128])])
    a <- evalIO (Policy.identity first)
    b <- evalIO (Policy.identity second)
    assert (a /= b)

reject :: String -> ByteString -> PropertyT IO ()
reject expectedError content = do
    path <- write content
    result <- evalIO (tryIOError (Policy.identity path))
    case result of
        Left problem -> assert (expectedError `isInfixOf` ioeGetErrorString problem)
        Right unexpected -> annotate unexpected >> failure

malformed :: PropertyT IO ()
malformed = do
    let tensor = "{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}"
        word = Bytes.pack [0, 0, 128, 63]
    reject "Truncated" "short"
    reject "header length" (prefix 100 <> "{}")
    reject "begin with an object" (framed " {}" Bytes.empty)
    reject "Duplicate" (framed ("{\"a\":" <> tensor <> ",\"a\":" <> tensor <> "}") word)
    reject "Duplicate" (framed "{\"a\":{\"dtype\":\"F32\",\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}" word)
    reject "nonempty" (framed "{}" Bytes.empty)
    reject "only strings" (framed "{\"__metadata__\":{\"value\":1}}" Bytes.empty)
    forM_ ["{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}", "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}"] $ \header ->
        reject "offsets" (framed header word)
    reject "data buffer" (framed ("{\"a\":" <> tensor <> "}") (word <> word))
    reject "profile" (framed "{\"a\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}" (Bytes.take 2 word))
    reject "Unsupported" (framed "{\"a\":{\"dtype\":\"F32\",\"shape\":[1.0],\"data_offsets\":[0,4]}}" word)

nonfinite :: PropertyT IO ()
nonfinite = forM_ [[0, 0, 128, 127], [0, 0, 128, 255], [1, 0, 128, 127]] $ \word ->
    reject "non-finite" (artifact [("invalid", [1], Bytes.pack word)])

checkpoint :: PropertyT IO ()
checkpoint = do
    root <- workspace
    let learner = TextBytes.replicate learnerLength 'x' <> "y"
        learnerLength = 70000
        identity = "0ddd6b7433742e7920ec0337e33d65cb6cd53541d7ab48eeb306d17f39a37cce"
    report <- checkpointResult expected identity
    evalIO (Bytes.writeFile (root </> "adapter.safetensors") (artifact tensors))
    evalIO (Bytes.writeFile (root </> "learner.pt") learner)
    evalIO (Worker.verifyCheckpoint root report) >>= (=== Right ())
    evalIO (Bytes.appendFile (root </> "learner.pt") "changed")
    changed <- evalIO (Worker.verifyCheckpoint root report)
    case changed of
        Left (Worker.LearnerMismatch claimed actual) -> claimed === identity >> assert (actual /= claimed)
        unexpected -> annotateShow unexpected >> failure
    evalIO (Bytes.writeFile (root </> "adapter.safetensors") (artifact (take 1 tensors)))
    mismatched <- evalIO (Worker.verifyCheckpoint root report)
    case mismatched of
        Left (Worker.PolicyMismatch claimed actual) -> claimed === expected >> assert (actual /= claimed)
        unexpected -> annotateShow unexpected >> failure

nonregular :: PropertyT IO ()
nonregular = do
    target <- write (artifact tensors)
    forM_ [createSymbolicLink target, createDirectory, (`createNamedPipe` ownerMode)] $ \create -> do
        root <- workspace
        let path = root </> "adapter.safetensors"
        evalIO (create path)
        result <- evalIO (tryIOError (Policy.identity path))
        case result of
            Left _ -> success
            Right unexpected -> annotate unexpected >> failure
  where
    ownerMode = 0o600
