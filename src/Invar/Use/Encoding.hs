{-# LANGUAGE OverloadedStrings #-}

module Invar.Use.Encoding (decodeContract, describeContract, criterionValue, standardValue, domainValue, methodValue) where

import Control.Monad (unless, (>=>))
import Data.Aeson (Value (..), encode, object, parseJSON, withObject, (.:), (.=))
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as ASCII
import Data.ByteString.Lazy qualified as Lazy
import Data.Char (digitToInt)
import Data.List.NonEmpty qualified as NonEmpty
import Data.Map.Strict qualified as Map
import Data.Ratio (denominator, numerator, (%))
import Data.Text qualified as Text
import Data.Text.Encoding qualified as Text
import Invar.Artifact qualified as Artifact
import Invar.Json qualified as Json
import Invar.Numerical qualified as N
import Invar.Policy qualified as Policy
import Invar.Spec.Decode qualified as Decode
import Invar.Spec.Domain qualified as D
import Invar.Spec.Encode qualified as Encode
import Invar.Spec.Measurement qualified as M
import Invar.Spec.Obligation qualified as Obligation
import Invar.Spec.Program qualified as P
import Invar.Spec.Syntax qualified as Syntax
import Invar.Use.Contract qualified as C

decodeContract :: ByteString -> Either String C.UseContract
decodeContract bytes = Json.decode bytes >>= parseEither parseContract

parseContract :: Value -> Parser C.UseContract
parseContract = withObject "use contract" $ \fields -> do
    Json.fields ["format", "purpose", "domain", "measurement", "reference", "candidate", "maximum_context", "criterion", "protocols", "reliance"] fields
    format <- fields .: "format"
    unless (format == ("invar-use-contract" :: String)) (fail "Unknown use contract format")
    reference <- fields .: "reference" >>= policy
    candidate <- fields .: "candidate" >>= policy
    protocols <- fields .: "protocols"
    Json.fields ["freeze", "isolation", "selection"] protocols
    C.UseContract
        <$> fields .: "purpose"
        <*> (fields .: "domain" >>= parseDomain)
        <*> (fields .: "measurement" >>= traverse parseMethod)
        <*> pure reference
        <*> pure candidate
        <*> fields .: "maximum_context"
        <*> (fields .: "criterion" >>= parseCriterion)
        <*> protocols .: "freeze"
        <*> protocols .: "isolation"
        <*> protocols .: "selection"
        <*> (fields .: "reliance" >>= traverse parseReliance)
  where
    policy :: Value -> Parser Policy.Description
    policy = either fail pure . Policy.decodeDescription . Lazy.toStrict . encode

parseCriterion :: Value -> Parser C.Criterion
parseCriterion = withObject "use criterion" $ \fields -> do
    Json.fields ["numerical", "loss", "invariance"] fields
    C.Criterion
        <$> (fields .: "numerical" >>= traverse parseNumerical)
        <*> (fields .: "loss" >>= traverse parseLoss)
        <*> (fields .: "invariance" >>= traverse parseInvariance)

parseInvariance :: Value -> Parser C.InvarianceRequirement
parseInvariance = withObject "invariance requirement" $ \fields -> do
    Json.fields ["relation", "executions", "rationale"] fields
    C.InvarianceRequirement <$> (fields .: "relation" >>= parseRelation) <*> fields .: "executions" <*> fields .: "rationale"

parseLoss :: Value -> Parser C.LossRequirement
parseLoss = withObject "bounded loss requirements" $ \fields -> do
    Json.fields ["reference_ceiling", "regression_ceiling", "standard"] fields
    C.LossRequirement
        <$> (fields .: "reference_ceiling" >>= parseBudget)
        <*> (fields .: "regression_ceiling" >>= parseBudget)
        <*> (fields .: "standard" >>= parseStandard)

parseBudget :: Value -> Parser C.Budget
parseBudget = withObject "justified budget" $ \fields -> do
    Json.fields ["limit", "rationale"] fields
    C.Budget <$> (fields .: "limit" >>= parseRational) <*> fields .: "rationale"

parseRational :: Value -> Parser Rational
parseRational = withObject "exact rational" $ \fields -> do
    Json.fields ["numerator", "denominator"] fields
    above <- fields .: "numerator"
    below <- fields .: "denominator"
    unless (below > 0) (fail "Expected positive rational denominator")
    pure (above % below)

parseNumerical :: Value -> Parser C.NumericalRequirement
parseNumerical = withObject "numerical requirement" $ \fields -> do
    Json.fields ["relation", "probe_steps", "rationale"] fields
    C.NumericalRequirement <$> (fields .: "relation" >>= parseRelation) <*> fields .: "probe_steps" <*> fields .: "rationale"

parseRelation :: Value -> Parser N.Relation
parseRelation = withObject "numerical relation" $ \fields -> do
    kind <- fields .: "kind"
    case kind :: String of
        "tokens" -> exact N.SameTokens fields
        "behavior-bits" -> exact N.SameBehaviorBits fields
        "termination" -> exact N.SameTermination fields
        "model-substitution" -> exact N.ModelSubstitution fields
        "prefix-log-ratio" -> bounded N.PrefixLogRatioWithin fields
        "path-log-ratio" -> bounded N.PathLogRatioWithin fields
        "reference-path-log-ratio" -> bounded (N.ScoredPathLogRatioWithin N.Reference) fields
        "candidate-path-log-ratio" -> bounded (N.ScoredPathLogRatioWithin N.Candidate) fields
        "full-vocabulary-kl" -> do
            Json.fields ["kind", "path", "direction", "budget"] fields
            side <- fields .: "path" >>= named [("reference", N.Reference), ("candidate", N.Candidate)]
            direction <- fields .: "direction" >>= named [("reference-candidate", N.ReferenceToCandidate), ("candidate-reference", N.CandidateToReference)]
            N.FullVocabularyKLWithin side direction <$> (fields .: "budget" >>= parseRational)
        _ -> fail "Unknown numerical relation"
  where
    exact relation fields = Json.fields ["kind"] fields >> pure relation
    bounded relation fields = Json.fields ["kind", "budget"] fields >> relation <$> (fields .: "budget" >>= parseRational)

parseStandard :: Value -> Parser C.Standard
parseStandard = withObject "evidence standard" $ \fields -> do
    kind <- fields .: "kind"
    case kind :: String of
        "finite_domain" -> Json.fields ["kind"] fields >> pure C.FiniteDomain
        "population_hoeffding" -> population C.HoeffdingPopulation fields
        "population_bernstein_mp2009" -> population C.EmpiricalBernsteinPopulation fields
        "conditional_derivation" -> do
            Json.fields ["kind", "assumptions"] fields
            C.ConditionalDerivation <$> (fields .: "assumptions" >>= traverse parseObligation)
        _ -> fail "Unknown evidence standard"
  where
    number fields key = fields .: key >>= parseRational
    population constructor fields = do
        Json.fields ["kind", "population", "unit_sampling", "replicate_sampling", "reference_alpha", "regression_alpha", "family_alpha"] fields
        constructor
            <$> ( C.Population
                    <$> fields .: "population"
                    <*> fields .: "unit_sampling"
                    <*> fields .: "replicate_sampling"
                    <*> number fields "reference_alpha"
                    <*> number fields "regression_alpha"
                    <*> number fields "family_alpha"
                )

parseReliance :: Value -> Parser C.Reliance
parseReliance = withObject "declared reliance" $ \fields -> do
    Json.fields ["premise", "authority", "basis_sha256"] fields
    kind <- fields .: "premise" >>= named [(show kind, kind) | kind <- [minBound .. maxBound]]
    identity <- fields .: "basis_sha256" >>= Json.identity >>= hexBytes
    C.Reliance kind <$> fields .: "authority" <*> pure identity

parseObligation :: Value -> Parser Obligation.Obligation
parseObligation = withObject "external obligation" $ \fields -> do
    Json.fields ["predicate", "specification_hex", "observation", "domain_hex", "binding_hex"] fields
    Obligation.Obligation
        <$> fields .: "predicate"
        <*> bytes fields "specification_hex"
        <*> fields .: "observation"
        <*> bytes fields "domain_hex"
        <*> bytes fields "binding_hex"
  where
    bytes fields key = fields .: key >>= hexBytes

named :: [(String, value)] -> Value -> Parser value
named choices value = do
    name <- parseJSON value
    maybe (fail ("Unknown value: " ++ name)) pure (lookup name choices)

hexBytes :: String -> Parser ByteString
hexBytes text = do
    unless (even (length text) && all (`elem` (['0' .. '9'] ++ ['a' .. 'f'])) text) (fail "Expected lowercase hexadecimal bytes")
    Bytes.pack <$> octets text
  where
    octets [] = pure []
    octets (a : b : remaining) = (fromIntegral (16 * digitToInt a + digitToInt b) :) <$> octets remaining
    octets _ = fail "Expected complete hexadecimal octets"

describeContract :: C.UseContract -> Value
describeContract contract =
    object
        [ "format" .= String "invar-use-contract"
        , "purpose" .= C.purpose contract
        , "domain" .= domainValue (C.declaredDomain contract)
        , "measurement" .= fmap methodValue (C.declaredMeasurement contract)
        , "reference" .= policyValue (C.referenceImplementation contract)
        , "candidate" .= policyValue (C.candidateImplementation contract)
        , "maximum_context" .= C.maximumContext contract
        , "criterion" .= criterionValue (C.criterion contract)
        , "protocols" .= object ["freeze" .= C.freezeProtocol contract, "isolation" .= C.isolationProtocol contract, "selection" .= C.selectionProtocol contract]
        , "reliance" .= [object ["premise" .= show (C.premise value), "authority" .= C.authority value, "basis_sha256" .= Artifact.hex (C.basis value)] | value <- C.reliance contract]
        ]

policyValue :: Policy.Description -> Value
policyValue policy =
    object
        [ "format" .= String "invar-policy-v1"
        , "model" .= Policy.model policy
        , "revision" .= Policy.revision policy
        , "adapter" .= Policy.adapter policy
        , "tokenizer" .= Policy.tokenizer policy
        , "base" .= Policy.base policy
        , "assembly" .= Policy.assembly policy
        ]

criterionValue :: C.Criterion -> Value
criterionValue criterion =
    object
        [ "numerical" .= fmap numericalValue (C.numericalRequirements criterion)
        , "loss" .= fmap lossValue (C.lossRequirement criterion)
        , "invariance" .= fmap invarianceValue (C.invarianceRequirements criterion)
        ]
  where
    numericalValue required = object ["relation" .= relationValue (C.relation required), "probe_steps" .= C.probeSteps required, "rationale" .= C.numericalRationale required]
    invarianceValue required = object ["relation" .= relationValue (C.invariant required), "executions" .= C.executions required, "rationale" .= C.invarianceRationale required]
    lossValue requested = object ["reference_ceiling" .= budgetValue (C.referenceCeiling requested), "regression_ceiling" .= budgetValue (C.regressionCeiling requested), "standard" .= standardValue (C.standard requested)]

budgetValue :: C.Budget -> Value
budgetValue budget = object ["limit" .= rational (C.limit budget), "rationale" .= C.rationale budget]

relationValue :: N.Relation -> Value
relationValue relation = case relation of
    N.SameTokens -> exact "tokens"
    N.SameBehaviorBits -> exact "behavior-bits"
    N.SameTermination -> exact "termination"
    N.ModelSubstitution -> exact "model-substitution"
    N.PrefixLogRatioWithin bound -> bounded "prefix-log-ratio" bound
    N.PathLogRatioWithin bound -> bounded "path-log-ratio" bound
    N.ScoredPathLogRatioWithin side bound -> bounded (case side of N.Reference -> "reference-path-log-ratio"; N.Candidate -> "candidate-path-log-ratio") bound
    N.FullVocabularyKLWithin side direction bound ->
        object
            [ "kind" .= String "full-vocabulary-kl"
            , "budget" .= rational bound
            , "path" .= String (case side of N.Reference -> "reference"; N.Candidate -> "candidate")
            , "direction" .= String (case direction of N.ReferenceToCandidate -> "reference-candidate"; N.CandidateToReference -> "candidate-reference")
            ]
  where
    exact kind = object ["kind" .= String kind]
    bounded kind bound = object ["kind" .= String kind, "budget" .= rational bound]

standardValue :: C.Standard -> Value
standardValue C.FiniteDomain = object ["kind" .= String "finite_domain"]
standardValue (C.HoeffdingPopulation population) = populationValue "population_hoeffding" population
standardValue (C.EmpiricalBernsteinPopulation population) = populationValue "population_bernstein_mp2009" population
standardValue (C.ConditionalDerivation assumptions) = object ["kind" .= String "conditional_derivation", "assumptions" .= map obligationValue assumptions]

populationValue :: Text.Text -> C.Population -> Value
populationValue kind population =
    object
        [ "kind" .= String kind
        , "population" .= C.populationName population
        , "unit_sampling" .= C.unitSampling population
        , "replicate_sampling" .= C.replicateSampling population
        , "reference_alpha" .= rational (C.referenceAlpha population)
        , "regression_alpha" .= rational (C.regressionAlpha population)
        , "family_alpha" .= rational (C.familyAlpha population)
        ]

obligationValue :: Obligation.Obligation -> Value
obligationValue value =
    object
        [ "predicate" .= Obligation.predicate value
        , "specification_hex" .= Artifact.hex (Obligation.specification value)
        , "observation" .= Obligation.observation value
        , "domain_hex" .= Artifact.hex (Obligation.domain value)
        , "binding_hex" .= Artifact.hex (Obligation.binding value)
        ]

rational :: Rational -> Value
rational value = object ["numerator" .= numerator value, "denominator" .= denominator value]

parseDomain :: Value -> Parser D.Domain
parseDomain = withObject "input domain" $ \fields -> do
    Json.fields ["name", "provenance_hex", "unit_definition", "inputs"] fields
    inputs <- fields .: "inputs" >>= traverse parseInput
    declared <- maybe (fail "Expected nonempty domain inputs") pure (NonEmpty.nonEmpty inputs)
    D.Domain <$> fields .: "name" <*> (fields .: "provenance_hex" >>= hexBytes) <*> fields .: "unit_definition" <*> pure declared
  where
    parseInput = withObject "declared input" $ \fields -> do
        Json.fields ["cohort", "key", "unit", "prompt", "tokens", "temperature", "seed", "parameters"] fields
        key <- D.Key <$> fields .: "cohort" <*> fields .: "key"
        parameters <- fields .: "parameters" >>= traverse (either fail pure . (Syntax.parse >=> Decode.value) . Text.encodeUtf8 . Text.pack)
        D.Input key
            <$> fields .: "unit"
            <*> fields .: "prompt"
            <*> fields .: "tokens"
            <*> (fields .: "temperature" >>= Json.finite)
            <*> fields .: "seed"
            <*> pure parameters

domainValue :: D.Domain -> Value
domainValue declared =
    object
        [ "name" .= D.domainName declared
        , "provenance_hex" .= Artifact.hex (D.provenance declared)
        , "unit_definition" .= D.unitDefinition declared
        , "inputs" .= fmap inputValue (D.declaredInputs declared)
        ]
  where
    inputValue input =
        object
            [ "cohort" .= D.cohort (D.inputKey input)
            , "key" .= D.task (D.inputKey input)
            , "unit" .= D.unitId input
            , "prompt" .= D.prompt input
            , "tokens" .= D.tokens input
            , "temperature" .= D.temperature input
            , "seed" .= D.seed input
            , "parameters" .= fmap (ASCII.unpack . Syntax.render . Encode.value) (D.parameters input)
            ]

parseMethod :: Value -> Parser M.Method
parseMethod = withObject "measurement method" $ \fields -> do
    Json.fields ["program_hex", "bindings", "sink", "specification", "lower", "upper", "orientation", "meaning"] fields
    bound <- fields .: "bindings" >>= traverse parseBinding
    let bindings = Map.fromList bound
    unless (Map.size bindings == length bound) (fail "Duplicate measurement source binding")
    spec <-
        M.MethodSpec
            <$> (fields .: "program_hex" >>= hexBytes)
            <*> pure bindings
            <*> fields .: "sink"
            <*> fields .: "specification"
            <*> (fields .: "lower" >>= parseRational)
            <*> (fields .: "upper" >>= parseRational)
            <*> (fields .: "orientation" >>= named [("increasing_loss", M.IncreasingLoss), ("decreasing_loss", M.DecreasingLoss)])
            <*> fields .: "meaning"
    either (fail . show) pure (M.prepare spec)

parseBinding :: Value -> Parser (P.Source, M.Binding)
parseBinding = withObject "measurement source binding" $ \fields -> do
    Json.fields ["source_kind", "source_name", "binding"] fields
    constructor <- fields .: "source_kind" >>= named [("semantic", P.Semantic), ("operational", P.Operational), ("logical_random", P.LogicalRandom)]
    source <- constructor <$> fields .: "source_name"
    value <-
        fields .: "binding"
            >>= withObject
                "bound field or parameter"
                ( \entry -> do
                    Json.fields ["kind", "name"] entry
                    kind <- entry .: "kind"
                    case kind :: String of
                        "field" -> M.ObservedField <$> (entry .: "name" >>= named [(show field, field) | field <- [minBound .. maxBound]])
                        "parameter" -> M.Parameter <$> entry .: "name"
                        _ -> fail "Unknown measurement input binding"
                )
    pure (source, value)

methodValue :: M.Method -> Value
methodValue method =
    object
        [ "program_hex" .= Artifact.hex (M.program spec)
        , "bindings" .= map bindingValue (Map.toAscList (M.bindings spec))
        , "sink" .= M.sink spec
        , "specification" .= M.outputSpecification spec
        , "lower" .= rational (M.lower spec)
        , "upper" .= rational (M.upper spec)
        , "orientation" .= String (case M.orientation spec of M.IncreasingLoss -> "increasing_loss"; M.DecreasingLoss -> "decreasing_loss")
        , "meaning" .= M.meaning spec
        ]
  where
    spec = M.specification method
    bindingValue (source, binding) =
        let (kind, name) = case source of P.Semantic value -> ("semantic", value); P.Operational value -> ("operational", value); P.LogicalRandom value -> ("logical_random", value)
         in object ["source_kind" .= String kind, "source_name" .= name, "binding" .= boundValue binding]
    boundValue (M.Parameter name) = object ["kind" .= String "parameter", "name" .= name]
    boundValue (M.ObservedField field) = object ["kind" .= String "field", "name" .= show field]
