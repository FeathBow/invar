{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Probability (validate) where

import Control.Monad (unless)
import Data.Aeson (Object, withObject, (.:))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Word (Word32)
import GHC.Float (castWord32ToFloat)
import Invar.Infer.Wire qualified as Binding
import Invar.Json qualified as Json
import Invar.Learn.Protocol qualified as P
import Invar.Spec.Invocation qualified as V

validate :: P.Result -> ByteString -> Either String ()
validate expected encoded = Json.decode encoded >>= parseEither (withObject "probability observation" (document expected))

document :: P.Result -> Object -> Parser ()
document expected fields = do
    exact ["format", "invocation", "request", "samples"] fields
    format <- fields .: "format"
    unless (format == ("invar-probabilities-v1" :: Text)) (fail "Unknown probability observation format")
    invocation <- fields .: "invocation"
    let completed = P.completion expected
        intended = Binding.invocationValue (V.completedBinding completed) (V.completedProgram completed)
    unless (invocation == intended) (fail "Probability observation invocation mismatch")
    request <- fields .: "request"
    unless (request == P.request expected) (fail "Probability observation request mismatch")
    observations <- fields .: "samples"
    withObject "probability request" (samples observations) request

samples :: [Object] -> Object -> Parser ()
samples observed request = do
    order <- request .: "order" :: Parser [Text]
    delivered <- request .: "samples" :: Parser [Object]
    named <- traverse (\item -> (,) <$> item .: "sample" <*> pure item) delivered
    names <- traverse (.: "sample") observed
    unless (names == order) (fail "Probability samples differ from logical order")
    let expected = Map.fromList named
    mapM_ (check expected) (zip names observed)
  where
    check expected (name, item) = case Map.lookup name expected of
        Nothing -> fail "Unknown probability sample"
        Just original -> sample original item

sample :: Object -> Object -> Parser ()
sample original fields = do
    exact (["sample", "dtype", "active"] ++ roles) fields
    dtype <- fields .: "dtype"
    unless (dtype == ("F32" :: Text)) (fail "Update probability observations must use FP32")
    behavior <- original .: "behavior_bits" :: Parser [Word32]
    actual <- fields .: "behavior"
    unless (actual == behavior) (fail "Behavior probability words differ from consumed input")
    mapM_ (vector (length behavior) fields) roles
    active <- fields .: "active" :: Parser [Bool]
    unless (active == replicate (length behavior) True) (fail "Update probability active mask mismatch")

roles :: [Key]
roles = ["behavior", "proximal", "reference", "current", "advantage"]

vector :: Int -> Object -> Key -> Parser ()
vector count fields role = do
    encoded <- fields .: role :: Parser [Word32]
    unless (length encoded == count && count > 0) (fail "Probability vector token count mismatch")
    let valid word = let number = castWord32ToFloat word in not (isNaN number || isInfinite number) && (role == "advantage" || number <= 0)
    unless (all valid encoded) (fail "Invalid probability or advantage floating-point words")

exact :: [Key] -> Object -> Parser ()
exact names fields = unless (Set.fromList (Fields.keys fields) == Set.fromList names) (fail "Unexpected probability observation fields")
