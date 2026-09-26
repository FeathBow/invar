{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

module UsePrepare (run, readDeclared) where

import Check (Check, andThen, problem, value)
import Check qualified
import Control.Monad (unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (Value (..), eitherDecodeStrict, encode, object, (.=))
import Data.Aeson.Key qualified as Key
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Char (toLower)
import Data.Foldable (toList)
import Data.List (isSuffixOf)
import Data.Ratio (denominator, numerator)
import Data.Text qualified as Text
import Envelope (Problem (..), refuse, succeed)
import Invar.Artifact qualified as Artifact
import Invar.Policy qualified as Policy
import Invar.Use qualified as U
import Invar.Use.Decimal qualified as Decimal
import Invar.Use.Measurement qualified as Measurement
import Invar.Use.Prepare qualified as Prepare
import Invar.Workload qualified as Workload
import Located qualified as L
import Options qualified as O
import System.Console.GetOpt (OptDescr)
import System.Directory (doesFileExist)
import System.FilePath (takeDirectory, (</>))
import UseExecution qualified as Execution

format :: String
format = "invar-use-prepared-v1"

data Reliance = Reliance {premise :: String, authority :: String, basis :: [FilePath]}

data Declaration = Declaration
    { workload :: FilePath
    , referencePolicy :: FilePath
    , candidatePolicy :: FilePath
    , purpose :: String
    , context :: Integer
    , criterion :: Value
    , protocols :: (String, String, String)
    , reliance :: [Reliance]
    , review :: [(String, String)]
    }

run :: [String] -> IO ()
run supplied = do
    fields <- either (\found -> refuse format [Problem "missing-argument" "argv" found]) pure (O.parse options supplied)
    declarationPath <- required fields "declaration"
    executionPath <- required fields "execution"
    output <- required fields "output"
    exists <- doesFileExist output
    when exists (refuse format [Problem "invalid-value" "argv:--output" (output ++ " already exists; prepare never overwrites a contract")])
    (declarationValue, declarationBytes) <- readDeclared "declaration" declarationPath
    (executionValue, executionBytes) <- readDeclared "execution" executionPath
    method <- either (\found -> refuse format [Problem "internal-error" "artifact:measurement" (show found)]) pure Decimal.method
    (declared, plan) <- either (\found -> refuse format (found ++ uncovered method declarationValue)) pure (Check.run ((,) <$> declaration declarationValue <*> Execution.decode executionValue))
    let base = takeDirectory declarationPath
    domain <- readWorkload (base </> workload declared)
    reference <- readPolicy "reference_policy" (base </> referencePolicy declared)
    candidate <- readPolicy "candidate_policy" (base </> candidatePolicy declared)
    reliedOn <- traverse (digestBasis base) (zip [0 ..] (reliance declared))
    let (freeze, isolation, selection) = protocols declared
        skeleton = U.UseContract (purpose declared) domain (Just method) reference candidate (fromInteger (context declared)) (U.Criterion [] Nothing []) freeze isolation selection [] []
        document = case U.describeContract skeleton of
            Object encoded -> Object (Fields.insert "criterion" (criterion declared) (Fields.insert "reliance" (toJSONList reliedOn) encoded))
            other -> other
        contractBytes = Lazy.toStrict (encode document)
    contract <- either (\found -> refuse format [Problem "invalid-value" "declaration:decisions" ("The declared decisions do not form a valid contract: " ++ found)]) pure (U.decodeContract contractBytes)
    let invalid = [Problem "invalid-value" "declaration:decisions" (show reason) | U.InvalidContract reason <- U.validate contract]
        expected = Prepare.expectedPremises contract
        declaredPremises = map fst reliedOn
        missing = [Problem "missing-reliance" "declaration:decisions.reliance" ("No reliance is declared for " ++ show kind ++ ", which this contract and execution plan raise") | kind <- expected, show kind `notElem` declaredPremises]
        report = Prepare.units contract
        scheduled = Execution.schedules (Prepare.inputCount report) (Prepare.candidateExecutions report) plan
    unless (null invalid && null missing) (refuse format (invalid ++ missing))
    either (refuse format) pure (Check.run scheduled)
    Bytes.writeFile output contractBytes
    succeed
        format
        "prepared"
        "The contract is fully constructed and every required decision and reliance is present."
        "That any premise is true, that the candidate is good, or that a run will succeed."
        [ "contract" .= output
        , "contract_sha256" .= digest contractBytes
        , "declaration_sha256" .= digest declarationBytes
        , "execution_sha256" .= digest executionBytes
        , "units"
            .= object
                [ "inputs" .= Prepare.inputCount report
                , "units" .= Prepare.unitCount report
                , "seeds_per_unit" .= [object ["seeds" .= seeds, "units" .= count] | (seeds, count) <- Prepare.seedsPerUnit report]
                , "candidate_executions" .= Prepare.candidateExecutions report
                , "note" .= ("Repeated executions do not add units." :: String)
                ]
        , "expected_premises" .= object ["execution_sha256" .= digest executionBytes, "kinds" .= map show expected]
        , "owner_review" .= [object ["at" .= at, "reads_as" .= text] | (at, text) <- review declared]
        ]
  where
    required fields name = maybe (refuse format [Problem "missing-argument" ("argv:--" ++ name) ("--" ++ name ++ " is required")]) pure (O.optional fields name)
    digest = Artifact.hex . SHA256.hash
    toJSONList values = toJSON' [object ["premise" .= kind, "authority" .= who, "basis_sha256" .= identity] | (kind, (who, identity)) <- values]
    toJSON' = Array . foldr (\entry rest -> pure entry <> rest) mempty

uncovered :: Measurement.Method -> Value -> [Problem]
uncovered method document =
    [ Problem "missing-reliance" "declaration:decisions.reliance" ("No reliance is declared for " ++ show kind ++ ", which this contract and execution plan raise")
    | kind <- Prepare.premisesFor (Prepare.Shape (method <$ quality) (quality >>= standard) (nonempty "numerical") (nonempty "invariance"))
    , show kind `notElem` declared
    ]
  where
    at = foldl (\found name -> found >>= \case Object fields -> Fields.lookup (Key.fromString name) fields; _ -> Nothing) (Just document)
    quality = case at ["decisions", "quality"] of Just (Object fields) -> Just fields; _ -> Nothing
    standard fields = case Fields.lookup "standard" fields of
        Just (Object chosen) -> case Fields.lookup "kind" chosen of
            Just (String "finite_domain") -> Just Prepare.Finite
            Just (String "population_hoeffding") -> Just Prepare.Hoeffding
            Just (String "population_bernstein_mp2009") -> Just Prepare.Bernstein
            _ -> Nothing
        _ -> Nothing
    nonempty name = case at ["decisions", "requirements", name] of
        Just (Array values) -> not (null values)
        _ -> False
    declared = case at ["decisions", "reliance"] of
        Just (Array entries) -> [Text.unpack premise | Object entry <- toList entries, Just (String premise) <- [Fields.lookup "premise" entry]]
        _ -> []

readDeclared :: String -> FilePath -> IO (Value, Bytes.ByteString)
readDeclared name path = do
    present <- doesFileExist path
    unless present (refuse format [Problem "artifact-missing" ("argv:--" ++ name) (path ++ " does not exist")])
    bytes <- Bytes.readFile path
    either (\found -> refuse format [Problem "artifact-invalid" ("artifact:" ++ path) found]) (\parsed -> pure (parsed, bytes)) (eitherDecodeStrict bytes)

readWorkload :: FilePath -> IO U.Domain
readWorkload path = do
    present <- doesFileExist path
    unless present (refuse format [Problem "artifact-missing" "declaration:artifacts.workload" (path ++ " does not exist")])
    document <- Bytes.readFile path >>= either (\found -> refuse format [Problem "artifact-invalid" ("artifact:" ++ path) found]) pure . Workload.decode
    either (\found -> refuse format [Problem "artifact-invalid" ("artifact:" ++ path) ("The workload does not form an exact-decimal domain: " ++ show found)]) pure (Decimal.domain document)

readPolicy :: String -> FilePath -> IO Policy.Description
readPolicy name path = do
    present <- doesFileExist path
    unless present (refuse format [Problem "artifact-missing" ("declaration:artifacts." ++ name) (path ++ " does not exist")])
    when (any (`isSuffixOf` map toLower path) [".pt", ".safetensors"]) (refuse format [Problem "artifact-role" ("declaration:artifacts." ++ name) "Expected the inference policy description policy.json written by invar policy, not a checkpoint file"])
    bytes <- Bytes.readFile path
    case eitherDecodeStrict bytes :: Either String Value of
        Right (Object fields) | Fields.lookup "format" fields /= Just (String "invar-policy-v1") -> refuse format [Problem "artifact-role" ("declaration:artifacts." ++ name) "Expected an invar-policy-v1 description; learner and initialization records carry the learner's assembly, not the inference implementation"]
        _ -> either (\found -> refuse format [Problem "artifact-invalid" ("artifact:" ++ path) found]) pure (Policy.decodeDescription bytes)

digestBasis :: FilePath -> (Int, Reliance) -> IO (String, (String, String))
digestBasis base (index, entry) = do
    let at = "declaration:decisions.reliance[" ++ show index ++ "].basis"
    files <-
        traverse
            ( \name -> do
                present <- doesFileExist (base </> name)
                unless present (refuse format [Problem "artifact-missing" at (name ++ " does not exist relative to the declaration")])
                (,) name <$> Artifact.identity "Reliance basis" (base </> name)
            )
            (basis entry)
    serialized <- either (\found -> refuse format [Problem "invalid-value" at found]) pure (Prepare.basisBytes files)
    pure (premise entry, (authority entry, Artifact.hex (SHA256.hash serialized)))

declaration :: Value -> Check Declaration
declaration document =
    L.object top document `andThen` \fields ->
        L.only top ["format", "artifacts", "decisions"] fields
            *> formatted fields
            *> ( L.field "missing-decision" top fields "artifacts" `andThen` L.object artifactsPath `andThen` \artifacts ->
                    L.only artifactsPath ["workload", "reference_policy", "candidate_policy"] artifacts
                        *> ( L.field "missing-decision" top fields "decisions" `andThen` L.object decisionsPath `andThen` \decisions ->
                                decided artifacts decisions
                           )
               )
  where
    top = L.root "declaration"
    artifactsPath = L.child top "artifacts"
    decisionsPath = L.child top "decisions"
    formatted fields = L.field "invalid-value" top fields "format" `andThen` L.text "invalid-value" (L.child top "format") `andThen` \found -> if found == "invar-use-declaration-v1" then value () else problem "invalid-value" (L.render (L.child top "format")) "Expected invar-use-declaration-v1"
    artifact artifacts name = L.field "artifact-missing" artifactsPath artifacts name `andThen` L.text "artifact-missing" (L.child artifactsPath name)
    decision = L.field "missing-decision" decisionsPath
    said decisions name = decision decisions name `andThen` L.text "missing-decision" (L.child decisionsPath name)
    decided artifacts decisions =
        L.only decisionsPath ["purpose", "scoring", "context_tokens", "quality", "requirements", "protocols", "reliance"] decisions
            *> scoring decisions
            *> ( build
                    <$> artifact artifacts "workload"
                    <*> artifact artifacts "reference_policy"
                    <*> artifact artifacts "candidate_policy"
                    <*> said decisions "purpose"
                    <*> (decision decisions "context_tokens" `andThen` L.positive (L.child decisionsPath "context_tokens"))
                    <*> (decision decisions "quality" `andThen` quality)
                    <*> (decision decisions "requirements" `andThen` requirements)
                    <*> (decision decisions "protocols" `andThen` protocolsOf)
                    <*> (decision decisions "reliance" `andThen` reliances)
               )
    build workloadPath referencePath candidatePath purposeText contextTokens (lossValue, lossReview) (numericalValues, invarianceValues, requirementReview) (freeze, isolation, selection) relied =
        Declaration
            workloadPath
            referencePath
            candidatePath
            purposeText
            (toInteger contextTokens)
            (object ["numerical" .= numericalValues, "loss" .= lossValue, "invariance" .= invarianceValues])
            (freeze, isolation, selection)
            relied
            ( [("declaration:decisions.purpose", "Use: " ++ purposeText), ("declaration:decisions.scoring", "Each response is scored by exact-decimal: loss 0 when its final line states the expected number, 1 otherwise or when truncated."), ("declaration:decisions.context_tokens", "Every prompt with its response limit must fit in " ++ show contextTokens ++ " tokens.")]
                ++ lossReview
                ++ requirementReview
                ++ [("declaration:decisions.reliance[" ++ show index ++ "]", authority entry ++ " vouches for " ++ premise entry ++ " on the basis of " ++ show (basis entry) ++ ".") | (index, entry) <- zip [0 :: Int ..] relied]
            )
    scoring decisions =
        let scoringPath = L.child decisionsPath "scoring"
         in decision decisions "scoring" `andThen` L.object scoringPath `andThen` \found ->
                L.only scoringPath ["method"] found
                    *> (L.field "missing-decision" scoringPath found "method" `andThen` L.text "missing-decision" (L.child scoringPath "method") `andThen` \name -> if name == "exact-decimal" then value () else problem "unsupported" (L.render (L.child scoringPath "method")) "exact-decimal is the only v0.1 scoring method")
    quality found =
        let qualityPath = L.child decisionsPath "quality"
         in L.object qualityPath found `andThen` \fields ->
                L.only qualityPath ["reference_ceiling", "regression_ceiling", "standard"] fields
                    *> ( ( \(referenceValue, referenceLimit, referenceReason) (regressionValue, regressionLimit, regressionReason) (standardValue, standardReview) ->
                            ( object ["reference_ceiling" .= referenceValue, "regression_ceiling" .= regressionValue, "standard" .= standardValue]
                            ,
                                [ ("declaration:decisions.quality.reference_ceiling", "The reference's mean task loss must be at most " ++ exact referenceLimit ++ ", because " ++ referenceReason ++ ".")
                                , ("declaration:decisions.quality.regression_ceiling", "The candidate may raise the mean task loss by at most " ++ exact regressionLimit ++ " over the reference, because " ++ regressionReason ++ ".")
                                , ("declaration:decisions.quality.standard", standardReview)
                                ]
                            )
                         )
                            <$> budget qualityPath fields "reference_ceiling"
                            <*> budget qualityPath fields "regression_ceiling"
                            <*> (L.field "missing-decision" qualityPath fields "standard" `andThen` standard (L.child qualityPath "standard"))
                       )
    budget parent fields name =
        let budgetPath = L.child parent name
         in L.field "missing-decision" parent fields name `andThen` L.object budgetPath `andThen` \entry ->
                L.only budgetPath ["limit", "rationale"] entry
                    *> ( (\limit reason -> (object ["limit" .= rationalValue limit, "rationale" .= reason], limit, reason))
                            <$> (L.field "missing-decision" budgetPath entry "limit" `andThen` L.rational (L.child budgetPath "limit"))
                            <*> (L.field "missing-decision" budgetPath entry "rationale" `andThen` L.text "missing-decision" (L.child budgetPath "rationale"))
                       )
    standard standardPath found =
        L.object standardPath found `andThen` \fields ->
            L.field "missing-decision" standardPath fields "kind" `andThen` L.text "missing-decision" (L.child standardPath "kind") `andThen` \kind -> case kind of
                "finite_domain" -> L.only standardPath ["kind"] fields *> value (object ["kind" .= kind], "The decision is the exact mean over the declared units; it makes no claim beyond them.")
                _
                    | kind `elem` ["population_hoeffding", "population_bernstein_mp2009"] ->
                        L.only standardPath ["kind", "population", "unit_sampling", "replicate_sampling", "reference_alpha", "regression_alpha", "family_alpha"] fields
                            *> ( ( \populationText unitText replicateText referenceAlpha regressionAlpha familyAlpha ->
                                    ( object ["kind" .= kind, "population" .= populationText, "unit_sampling" .= unitText, "replicate_sampling" .= replicateText, "reference_alpha" .= rationalValue referenceAlpha, "regression_alpha" .= rationalValue regressionAlpha, "family_alpha" .= rationalValue familyAlpha]
                                    , "Upper confidence bounds (" ++ kind ++ ") hold for the population '" ++ populationText ++ "', with alphas " ++ exact referenceAlpha ++ " for the reference loss and " ++ exact regressionAlpha ++ " for the increase, within a family alpha of " ++ exact familyAlpha ++ "."
                                    )
                                 )
                                    <$> described fields standardPath "population"
                                    <*> described fields standardPath "unit_sampling"
                                    <*> described fields standardPath "replicate_sampling"
                                    <*> alpha fields standardPath "reference_alpha"
                                    <*> alpha fields standardPath "regression_alpha"
                                    <*> alpha fields standardPath "family_alpha"
                               )
                    | otherwise -> problem "unsupported" (L.render (L.child standardPath "kind")) "Expected finite_domain, population_hoeffding or population_bernstein_mp2009"
    described fields parent name = L.field "missing-decision" parent fields name `andThen` L.text "missing-decision" (L.child parent name)
    alpha fields parent name = L.field "missing-decision" parent fields name `andThen` L.rational (L.child parent name)
    requirements found =
        let requirementsPath = L.child decisionsPath "requirements"
         in L.object requirementsPath found `andThen` \fields ->
                L.only requirementsPath ["numerical", "invariance"] fields
                    *> ( (\(numericalValues, numericalReview) (invarianceValues, invarianceReview) -> (numericalValues, invarianceValues, numericalReview ++ invarianceReview))
                            <$> (L.field "missing-decision" requirementsPath fields "numerical" `andThen` entries (L.child requirementsPath "numerical") numerical)
                            <*> (L.field "missing-decision" requirementsPath fields "invariance" `andThen` entries (L.child requirementsPath "invariance") invariance)
                       )
    entries entriesPath single found = L.list entriesPath found `andThen` \values -> (\checked -> (map fst checked, map snd checked)) <$> traverse (\(index, entry) -> single (L.item entriesPath index) entry) (zip [0 ..] values)
    numerical entryPath found =
        L.object entryPath found `andThen` \fields ->
            L.only entryPath ["relation", "probe_steps", "rationale"] fields
                *> ( (\relation steps reason -> (object ["relation" .= relation, "probe_steps" .= steps, "rationale" .= reason], (L.render entryPath, "Admission requires the numerical relation " ++ relationName relation ++ " for every input, because " ++ reason ++ ".")))
                        <$> L.field "missing-decision" entryPath fields "relation"
                        <*> maybe (value (Array mempty)) value (L.optionalField fields "probe_steps")
                        <*> (L.field "missing-decision" entryPath fields "rationale" `andThen` L.text "missing-decision" (L.child entryPath "rationale"))
                   )
    invariance entryPath found =
        L.object entryPath found `andThen` \fields ->
            L.only entryPath ["relation", "executions", "rationale"] fields
                *> ( (\relation count reason -> (object ["relation" .= relation, "executions" .= count, "rationale" .= reason], (L.render entryPath, "Across " ++ show count ++ " candidate executions under different schedules, " ++ relationName relation ++ " must be identical, because " ++ reason ++ ".")))
                        <$> L.field "missing-decision" entryPath fields "relation"
                        <*> (L.field "missing-decision" entryPath fields "executions" `andThen` L.natural (L.child entryPath "executions") `andThen` \count -> if count >= 2 then value count else problem "invalid-value" (L.render (L.child entryPath "executions")) "Invariance needs at least two executions")
                        <*> (L.field "missing-decision" entryPath fields "rationale" `andThen` L.text "missing-decision" (L.child entryPath "rationale"))
                   )
    protocolsOf found =
        let protocolsPath = L.child decisionsPath "protocols"
         in L.object protocolsPath found `andThen` \fields ->
                L.only protocolsPath ["freeze", "isolation", "selection"] fields
                    *> ((,,) <$> described fields protocolsPath "freeze" <*> described fields protocolsPath "isolation" <*> described fields protocolsPath "selection")
    reliances found =
        let reliancePath = L.child decisionsPath "reliance"
         in L.list reliancePath found `andThen` \values -> traverse (\(index, entry) -> relianceEntry (L.item reliancePath index) entry) (zip [0 ..] values)
    relianceEntry entryPath found =
        L.object entryPath found `andThen` \fields ->
            L.only entryPath ["premise", "authority", "basis"] fields
                *> ( Reliance
                        <$> described fields entryPath "premise"
                        <*> described fields entryPath "authority"
                        <*> (L.field "missing-decision" entryPath fields "basis" `andThen` L.list (L.child entryPath "basis") `andThen` \names -> if null names then problem "missing-decision" (L.render (L.child entryPath "basis")) "A basis needs at least one file" else traverse (\(index, name) -> L.text "invalid-value" (L.item (L.child entryPath "basis") index) name) (zip [0 ..] names))
                   )
    relationName (Object fields) = case Fields.lookup (Key.fromString "kind") fields of
        Just (String kind) -> show kind
        _ -> "(unnamed)"
    relationName _ = "(unnamed)"

rationalValue :: Rational -> Value
rationalValue number = object ["numerator" .= numerator number, "denominator" .= denominator number]

exact :: Rational -> String
exact number = show (numerator number) ++ "/" ++ show (denominator number)

options :: [OptDescr (String, String)]
options = O.descriptions [("declaration", "Use declaration (invar-use-declaration-v1)"), ("execution", "Execution plan (invar-use-execution-v1)"), ("output", "Path of the contract to write; must not exist")]
