{-# LANGUAGE OverloadedStrings #-}

module Invar.Qualification.Profile (Role (..), Profile, decode, verify, policy, proof, name) where

import Control.Monad (unless, when)
import Data.Aeson (Object, Value (..), encode, withObject, (.:))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Lazy qualified as Lazy
import Data.List (sort)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Invar.Infer qualified as Infer
import Invar.Json qualified as Json
import Invar.Learn.Wire qualified as Learning
import Invar.Spec.Evidence qualified as Evidence
import Invar.Spec.Invocation qualified as Invocation
import Invar.Spec.Qualification qualified as Q

data Role = Inference | Learning
    deriving (Eq, Show)

data Region = Region Q.Definition Bool
data Profile = Profile Role (String, String, String) Q.Policy [Region] Bool

decode :: Role -> ByteString -> Either String Profile
decode selected encoded = do
    supplied <- Json.decode encoded >>= parseEither (withObject "qualification document" document)
    case filter ((== selected) . role) supplied of
        [matched] -> Right matched
        _ -> Left "Qualification document must contain exactly one profile for the requested numerical role"
  where
    document fields = do
        Json.fields ["format", "profiles"] fields
        format <- fields .: "format"
        unless (format == ("invar-numerical-qualification/v1" :: Text)) (fail "Unsupported qualification document format")
        values <- fields .: "profiles" >>= traverse (withObject "qualification profile" profile)
        unless (not (null values) && length values == length (uniqueRoles values)) (fail "Qualification profiles must have distinct numerical roles")
        pure values
    uniqueRoles = Map.keys . Map.fromList . map (\item -> (name item, ()))

profile :: Object -> Parser Profile
profile fields = do
    Json.fields ["role", "tokenizer", "base", "assembly", "target", "regions", "composition"] fields
    selected <- fields .: "role" >>= parseRole
    materialization <- (,,) <$> identity "tokenizer" <*> identity "base" <*> identity "assembly"
    target <- fields .: "target" >>= withObject "whole-operation refinement" definition
    regions <- fields .: "regions" >>= traverse (withObject "numerical region" region)
    compose <- fields .: "composition" >>= method
    unless (Q.predicate target == targetPredicate selected) (fail "Qualification target differs from the requested native operation")
    unless (sort [Q.predicate item | Region item _ <- regions] == sort (required selected)) (fail "Qualification profile has an incomplete or different numerical region inventory")
    checked <- either (fail . show) pure (Q.policy target [item | Region item _ <- regions] (Q.Premises [item | Region item True <- regions] compose))
    pure (Profile selected materialization checked regions compose)
  where
    identity key = fields .: key >>= Json.identity
    region values = do
        Json.fields ["predicate", "reference", "observation", "method"] values
        defined <- definition (Fields.delete "method" values)
        Region defined <$> (values .: "method" >>= method)

definition :: Object -> Parser Q.Definition
definition fields = do
    Json.fields ["predicate", "reference", "observation"] fields
    predicate <- fields .: "predicate" >>= nonempty
    observed <- fields .: "observation" >>= nonempty
    reference <- fields .: "reference" >>= withObject "executable numerical reference" executableReference
    pure (Q.Definition predicate (Lazy.toStrict (encode reference)) observed)
  where
    executableReference values = do
        Json.fields ["name", "artifact", "entry"] values
        _ <- values .: "name" >>= nonempty
        _ <- values .: "artifact" >>= Json.identity
        _ <- values .: "entry" >>= nonempty
        pure (Object values)

nonempty :: String -> Parser String
nonempty value = do
    when (null value || '\0' `elem` value) (fail "Expected nonempty qualification reference or observation text")
    pure value

method :: Text -> Parser Bool
method "external-assumption" = pure True
method "unknown" = pure False
method _ = fail "Expected an external-assumption or unknown evidence method; no unchecked proof label is admitted"

parseRole :: Text -> Parser Role
parseRole "inference" = pure Inference
parseRole "learning" = pure Learning
parseRole _ = fail "Expected an inference or learning qualification role"

role :: Profile -> Role
role (Profile selected _ _ _ _) = selected

name :: Profile -> String
name selected = case role selected of
    Inference -> "inference"
    Learning -> "learning"

policy :: Profile -> Q.Policy
policy (Profile _ _ selected _ _) = selected

targetPredicate :: Role -> String
targetPredicate Inference = "categorical-inference-refinement/v1"
targetPredicate Learning = "grpo-update-refinement/v1"

required :: Role -> [String]
required selected = map (++ "-correspondence/v1") (shared ++ specific)
  where
    shared = ["materialization", "model-forward", "observation"]
    specific = case selected of
        Inference -> ["tokenization", "sampling"]
        Learning -> ["scalar-objective", "model-differentiation", "optimizer-restoration"]

verify :: Profile -> Invocation.Intention -> Either String ()
verify (Profile selected expected _ _ _) intended = do
    actual <- case selected of
        Inference -> do
            requested <- first show (Infer.fromEmission (Invocation.intendedEmission intended))
            pure (Infer.tokenizer requested, Infer.base requested, Infer.assembly requested)
        Learning -> do
            requested <- first show (Learning.lower (Invocation.intendedEmission intended))
            parseEither (withObject "bound learning request" materialization) requested
    unless (actual == expected) (Left "Qualification profile differs from the actual tokenizer base or assembly")
  where
    materialization fields = (,,) <$> (fields .: "tokenizer" >>= Json.identity) <*> (fields .: "base" >>= Json.identity) <*> (fields .: "assembly" >>= Json.identity)

proof :: Profile -> Q.Request -> (Evidence.Graph, Evidence.EvidenceId)
proof (Profile _ _ _ regions compose) requested = (Map.fromList (observations ++ derived), root)
  where
    claims = Q.regionClaims requested
    evidence = True : [assumed | Region _ assumed <- regions]
    indices = [Evidence.EvidenceId index | (index, _) <- zip [0 ..] claims]
    observations =
        [ (index, Evidence.Node claim (if position == (0 :: Int) then Evidence.Compare else Evidence.Assume))
        | (position, (index, claim, present)) <- zip [0 ..] (zip3 indices claims evidence)
        , present
        ]
    group = Evidence.EvidenceId (fromIntegral (length claims))
    implication = Evidence.EvidenceId (fromIntegral (length claims) + 1)
    root = Evidence.EvidenceId (fromIntegral (length claims) + 2)
    derived =
        [(group, Evidence.Node (Q.requiredRegions requested) (Evidence.Conjoin indices))]
            ++ [(implication, Evidence.Node (Q.composition requested) Evidence.Assume) | compose]
            ++ [(root, Evidence.Node (Q.conclusion requested) (Evidence.Apply implication group))]
