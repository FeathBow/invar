{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

module Invar.Score.Program (Path (..), prepare) where

import Data.ByteString (ByteString)
import Data.Char (ord)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Invar.Construct qualified as C
import Invar.Infer qualified as Infer
import Invar.Infer.Schema qualified as Schema
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Program qualified as P
import Invar.Spec.Value (Scalar (..), Value (..))
import Numeric.Natural (Natural)

data Path = Path {sourceLog :: String, inspectionDigest :: String, prefix :: [Natural], response :: [Natural], probeSteps :: [Natural]}

type Tokens = C.Record '[ '("source_log", [Natural]), '("source_inspection", [Natural]), '("prefix", [Natural]), '("response", [Natural])]

prepare :: V.Binding -> Infer.Request -> Path -> Either String (ByteString, V.Runtime)
prepare bound requested selected = do
    checked <- either (Left . show) Right (program (not (null (probeSteps selected))))
    E.Emission _ _ emitted <- either (Left . show) Right (Infer.emission requested)
    inputs <- case emitted of
        Record fields -> do
            request <- required "request" fields
            policy <- required "policy" fields
            seed <- required "seed" fields
            let common = Map.fromList [(P.Semantic "request", request), (P.Semantic "policy", policy), (P.LogicalRandom "sample", seed), (P.Semantic "path", pathValue selected)]
            pure (if null (probeSteps selected) then common else Map.insert (P.Semantic "probe") (Sequence (map (Atom . Token) (probeSteps selected))) common)
        _ -> Left "Expected the checked inference input record"
    runtime <- either (Left . show) Right (V.prepare (V.Selection (V.boundCall bound) inputs) (V.start checked 0))
    pure (A.bytes checked, runtime)
  where
    required name = maybe (Left ("Missing inference input: " ++ name)) Right . Map.lookup name

program :: Bool -> Either C.BuildError A.Checked
program fullVocabulary
    | fullVocabulary = C.compile semantics [C.emit @"score" @"cached-distribution-probe/v1" @'[ 'C.Semantic "request", 'C.Semantic "policy", 'C.Semantic "path", 'C.Semantic "probe", 'C.LogicalRandom "sample"] (C.record (C.field @"probe_steps" (C.source @('C.Semantic "probe") @[Natural]) fields))]
    | otherwise = C.compile semantics [C.emit @"score" @"cached-path-score/v1" @'[ 'C.Semantic "request", 'C.Semantic "policy", 'C.Semantic "path", 'C.LogicalRandom "sample"] (C.record fields)]
  where
    fields = C.field @"policy" (C.source @('C.Semantic "policy") @(C.Record '[ '("artifact", [Natural]), '("profile", [Natural])])) (C.field @"request" (C.source @('C.Semantic "request") @Schema.Inputs) (C.field @"path" (C.source @('C.Semantic "path") @Tokens) (C.field @"seed" (C.numberSource @('C.LogicalRandom "sample")) C.emptyFields)))
    pathType = P.RecordType (Map.fromList [(name, P.SequenceType P.TokenType) | name <- ["source_log", "source_inspection", "prefix", "response"]])
    sources = Map.fromList ([(P.Semantic "request", Schema.inputs), (P.Semantic "policy", Schema.policy), (P.Semantic "path", pathType), (P.LogicalRandom "sample", P.NumberType)] ++ [(P.Semantic "probe", P.SequenceType P.TokenType) | fullVocabulary])
    output = P.RecordType (Map.fromList ([("policy", Schema.policy), ("request", Schema.inputs), ("path", pathType), ("seed", P.NumberType)] ++ [("probe_steps", P.SequenceType P.TokenType) | fullVocabulary]))
    sink = P.Sink (if fullVocabulary then "cached-distribution-probe/v1" else "cached-path-score/v1") output (Map.keysSet sources) Set.empty
    semantics = E.Semantics (P.Schema sources Map.empty (Map.singleton "score" sink)) Map.empty

pathValue :: Path -> Value Natural
pathValue selected = Record (Map.fromList [("source_log", text (sourceLog selected)), ("source_inspection", text (inspectionDigest selected)), ("prefix", tokens (prefix selected)), ("response", tokens (response selected))])
  where
    text = tokens . map (fromIntegral . ord)
    tokens = Sequence . map (Atom . Token)
