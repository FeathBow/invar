{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Report (Report, admit, admitFrames, consumedPolicy, bindingValue, paired, sameInput, describe, invocation, request, result, output, gradient, artifact, logDigest) where

import Control.Monad (unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Object, Value (Object, String), object, toJSON, withObject, (.:), (.=))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text (Text)
import Data.Text qualified as Text
import Invar.Artifact qualified as Artifact
import Invar.Infer.Framing qualified as Framing
import Invar.Json qualified as Json
import Invar.Learn.Request qualified as Request
import Numeric.Natural (Natural)

data Report = Report String Value Request.Request (Object, ByteString) String

logDigest :: Report -> String
logDigest (Report digest _ _ _ _) = digest

invocation :: Report -> Value
invocation (Report _ bound _ _ _) = bound

request :: Report -> Value
request (Report _ _ consumed _ _) = Request.value consumed

result :: Report -> Value
result (Report _ _ _ (fields, _) _) = Object fields

output :: Report -> ByteString
output (Report _ _ _ (_, encoded) _) = encoded

gradient :: Report -> String
gradient (Report _ _ _ _ digest) = digest

artifact :: Key -> Report -> Either String String
artifact name (Report _ _ _ (fields, _) _) = parseEither (\value -> value .: name >>= Json.identity) fields

admit :: Natural -> ByteString -> Either String Report
admit call encoded = do
    events <- traverse (\line -> (,line) <$> (Json.decode line >>= parseEither (withObject "execution log event" pure))) (Bytes.lines encoded)
    admitEvents call (Artifact.hex (SHA256.hash encoded)) events

admitFrames :: Natural -> [Framing.Frame] -> Either String Report
admitFrames call frames =
    let digest = Artifact.hex (SHA256.hash (Framing.encode frames))
     in length digest `seq` admitEvents call digest [(Framing.fields frame, Framing.raw frame) | frame <- frames]

admitEvents :: Natural -> String -> [(Object, ByteString)] -> Either String Report
admitEvents call digest events = parseEither (observation digest) (filter (selected . fst) events)
  where
    selected event = case (Fields.lookup "stage" event, Fields.lookup "binding" event) of
        (Just (String stage), Just (Object bound))
            | stage `elem` ["consumed", "result"] ->
                Fields.lookup "call" bound == Just (toJSON call)
        _ -> False

observation :: String -> [(Object, ByteString)] -> Parser Report
observation digest [(consumed, _), (returned, encoded)] = do
    stages <- traverse (.: "stage") [consumed, returned]
    unless (stages == (["consumed", "result"] :: [Text])) (fail "Expected one consumed/result pair for the selected call")
    bound <- consumed .: "binding" >>= binding
    program <- consumed .: "program"
    when (Text.null program) (fail "Expected nonempty program text")
    actual <- returned .: "binding"
    unless (actual == bound) (fail "Update result binding differs from consumption")
    input <- consumed .: "request" >>= Request.parse
    actualRequest <- returned .: "request"
    unless (actualRequest == Request.value input) (fail "Update result input differs from consumption")
    identity <- returned .: "gradients" >>= Json.identity
    pure (Report digest (object ["binding" .= bound, "program" .= program]) input (returned, encoded) identity)
observation _ _ = fail "Expected one consumed/result pair for the selected call"

binding :: Value -> Parser Value
binding = withObject "invocation binding" $ \fields -> do
    Json.fields ["call", "attempt", "instance"] fields
    mapM_ (\name -> fields .: name :: Parser Natural) ["call", "attempt", "instance"]
    pure (Object fields)

paired :: Report -> Report -> Either String ()
paired left right = do
    same <- sameInput left right
    unless same (Left "Compared updates must have the same program and consumed numerical input")

sameInput :: Report -> Report -> Either String Bool
sameInput (Report _ left first _ _) (Report _ right second _ _) = do
    leftProgram <- parseEither (withObject "invocation" (.: "program")) left :: Either String Text
    rightProgram <- parseEither (withObject "invocation" (.: "program")) right
    pure (leftProgram == rightProgram && Request.logical first == Request.logical second)

describe :: Report -> Value
describe report@(Report digest _ _ _ _) = object ["log_digest" .= digest, "invocation" .= invocation report, "request" .= request report, "digest" .= gradient report, "result" .= result report]

consumedPolicy :: Report -> Either String String
consumedPolicy = parseEither (withObject "request" (.: "policy")) . request

bindingValue :: Report -> Either String Value
bindingValue = parseEither (withObject "invocation" (.: "binding")) . invocation
