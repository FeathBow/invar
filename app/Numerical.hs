{-# LANGUAGE OverloadedStrings #-}

module Numerical (run, readPair, readRun) where

import Control.Monad (unless, when)
import Data.Aeson (Value (..), eitherDecodeStrict, encode, object, (.=))
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as TextBytes
import Data.ByteString.Lazy.Char8 qualified as Lazy
import Data.Maybe (isJust, isNothing)
import InferenceInput qualified
import Invar.Infer qualified as Infer
import Invar.Infer.Observation qualified as Inference
import Invar.Numerical qualified as N
import Invar.Score qualified as Score
import Numeric.Natural (Natural)
import Options qualified as O
import System.Console.GetOpt (OptDescr, usageInfo)
import System.Exit (die)

run :: [String] -> IO ()
run ["--help"] = putStrLn usage
run supplied = do
    fields <- either die pure (O.parse options supplied)
    relation <- either die pure (selectedRelation fields)
    suppliedPair <- pair fields
    observed <- either (die . show) pure (N.observe suppliedPair)
    let found = N.establish (N.Claim (N.scope observed) relation) observed
    Lazy.putStrLn (encode (object ["observation" .= N.describe observed, "finding" .= N.describeFinding found]))

readPair :: [String] -> IO N.BoundRun
readPair supplied = either die pure (O.parse pairOptions supplied) >>= pair

readRun :: [String] -> IO N.Run
readRun supplied = either die pure (O.parse runOptions supplied) >>= (`input` "")

runOptions :: [OptDescr (String, String)]
runOptions = InferenceInput.options ++ O.descriptions [("log", "Complete standalone or finite-batch inference stdout"), ("exit-code", "Independently recorded process exit status")]

pair :: O.Fields -> IO N.BoundRun
pair fields = do
    left <- input fields "reference-"
    right <- input fields "candidate-"
    referenceScores <- score fields N.Reference (left, right)
    candidateScores <- score fields N.Candidate (right, left)
    probes <- probeInputs fields (left, right)
    pure (N.ProbedRun left right (referenceScores ++ candidateScores) probes)

input :: O.Fields -> String -> IO N.Run
input fields prefix = do
    requested <- either die pure (InferenceInput.requestWith prefix fields)
    planned <- either (die . show) pure (Infer.prepare requested)
    binding <- either die pure (InferenceInput.bindingWith prefix fields)
    status <- either die pure (O.numeric fields (prefix ++ "exit-code"))
    path <- either die pure (O.required fields (prefix ++ "log"))
    bytes <- Bytes.readFile path
    pure (N.Run planned binding status bytes)

score :: O.Fields -> N.Side -> (N.Run, N.Run) -> IO [(N.Side, Score.Report)]
score fields side (source, target) = do
    probes <- traverse (either die pure . eitherDecodeStrict . TextBytes.pack) (O.optional fields (prefix ++ "probe-steps"))
    maybe [] (\measured -> [(side, measured)]) <$> scoreInput fields prefix (source, target, probes)
  where
    prefix = case side of N.Reference -> "reference-score-"; N.Candidate -> "candidate-score-"

scoreInput :: O.Fields -> String -> (N.Run, N.Run, Maybe [Natural]) -> IO (Maybe Score.Report)
scoreInput fields prefix (source, target, probes) = case O.optional fields (prefix ++ "log") of
    Nothing -> do
        when (any (\key -> isJust (O.optional fields (prefix ++ key))) ["call", "attempt", "instance", "exit-code", "probe-steps"]) (die ("Missing " ++ prefix ++ "log"))
        pure Nothing
    Just path -> do
        original <- either die pure (Inference.admit (N.planned source) (N.binding source) (N.logBytes source))
        base <- either (die . show) pure (Score.prepare (N.exitCode source) original (N.planned target))
        planned <- either (die . show) pure (maybe (Right base) (`Score.withProbe` base) probes)
        binding <- either die pure (InferenceInput.bindingWith prefix fields)
        call <- either (die . show) pure (Score.bind binding planned)
        status <- either die pure (O.numeric fields (prefix ++ "exit-code"))
        bytes <- Bytes.readFile path
        measured <- either (die . show) pure (Score.admit call status bytes)
        pure (Just measured)

probeInputs :: O.Fields -> (N.Run, N.Run) -> IO [N.Probe]
probeInputs fields (left, right)
    | not present = do
        unless (isNothing (O.optional fields "probe-steps")) (die "Probe steps require probe execution inputs")
        pure []
    | otherwise = do
        side <- either die pure (probeSide fields)
        encoded <- either die pure (O.required fields "probe-steps")
        steps <- either die pure (eitherDecodeStrict (TextBytes.pack encoded))
        let source = case side of N.Reference -> left; N.Candidate -> right
        before <- scoreInput fields "reference-probe-" (source, left, Just steps)
        after <- scoreInput fields "candidate-probe-" (source, right, Just steps)
        pure ([N.Probe side N.Reference report | Just report <- [before]] ++ [N.Probe side N.Candidate report | Just report <- [after]])
  where
    present = any (isJust . O.optional fields) [prefix ++ suffix | prefix <- ["reference-probe-", "candidate-probe-"], suffix <- ["log", "exit-code", "call", "attempt", "instance"]]

probeSide :: O.Fields -> Either String N.Side
probeSide fields = do
    selected <- O.required fields "probe-path"
    case selected of
        "reference" -> Right N.Reference
        "candidate" -> Right N.Candidate
        _ -> Left "Probe path must be reference or candidate"

selectedRelation :: O.Fields -> Either String N.Relation
selectedRelation fields = do
    selected <- O.required fields "relation"
    case selected of
        "tokens" -> exact N.SameTokens
        "behavior-bits" -> exact N.SameBehaviorBits
        "termination" -> exact N.SameTermination
        "prefix-log-ratio" -> N.PrefixLogRatioWithin <$> budget
        "path-log-ratio" -> N.PathLogRatioWithin <$> budget
        "reference-path-log-ratio" -> N.ScoredPathLogRatioWithin N.Reference <$> budget
        "candidate-path-log-ratio" -> N.ScoredPathLogRatioWithin N.Candidate <$> budget
        "kl-reference-candidate" -> N.FullVocabularyKLWithin <$> probeSide fields <*> pure N.ReferenceToCandidate <*> budget
        "kl-candidate-reference" -> N.FullVocabularyKLWithin <$> probeSide fields <*> pure N.CandidateToReference <*> budget
        "model-substitution" -> exact N.ModelSubstitution
        _ -> Left "Unknown numerical relation; see compare numerical --help"
  where
    exact relation = do
        unless (isNothing (O.optional fields "budget")) (Left "This relation does not consume a budget")
        pure relation
    budget = do
        supplied <- O.required fields "budget"
        decoded <- eitherDecodeStrict (TextBytes.pack supplied)
        case decoded of
            Number value -> Right (toRational value)
            _ -> Left "Expected a finite JSON number as the budget"

options :: [OptDescr (String, String)]
options =
    pairOptions
        ++ O.descriptions
            [ ("relation", "tokens | behavior-bits | termination | prefix-log-ratio | path-log-ratio | reference-path-log-ratio | candidate-path-log-ratio | kl-reference-candidate | kl-candidate-reference | model-substitution")
            , ("budget", "Exact decimal bound; log-ratio bounds are absolute values in natural-log units")
            ]

pairOptions :: [OptDescr (String, String)]
pairOptions =
    concatMap side ["reference-", "candidate-"]
        ++ concatMap scored ["reference-score-", "candidate-score-"]
        ++ concatMap probed ["reference-probe-", "candidate-probe-"]
        ++ O.descriptions
            [ ("probe-path", "reference | candidate: source path for both full-vocabulary probes and the KL claim")
            , ("probe-steps", "Frozen increasing JSON array of response steps, required when supplying probe logs")
            ]
  where
    side prefix =
        InferenceInput.optionsWith prefix
            ++ O.descriptions
                [ (prefix ++ "log", "Complete standalone or finite-batch inference stdout")
                , (prefix ++ "exit-code", "Independently recorded process exit status")
                ]
    scored prefix =
        O.descriptions
            ( [ (prefix ++ "log", "Complete score of this side's path through the opposite implementation")
              , (prefix ++ "probe-steps", "Optional nonempty increasing JSON array of response steps from this score's declared full-vocabulary call")
              ]
                ++ executionIdentity prefix
            )
    probed prefix =
        O.descriptions ((prefix ++ "log", "Complete full-vocabulary score through this implementation on the selected probe path") : executionIdentity prefix)
    executionIdentity prefix =
        [ (prefix ++ "exit-code", "Independently recorded score process exit status")
        , (prefix ++ "call", "Actual score call identity")
        , (prefix ++ "attempt", "Actual score attempt identity")
        , (prefix ++ "instance", "Actual score activation instance")
        ]

usage :: String
usage = usageInfo "Usage: invar compare numerical OPTIONS\nCompare finite paired inference observations using declared calls and actual log snapshots.\nOptional reference-score-* and candidate-score-* inputs admit full-path cross-scores.\nDeclare each full-vocabulary score's own steps with its score-probe-steps option; omission declares a plain score.\nEach selected-path ratio is source minus opposite target on that source's path.\nKL requires --probe-path; reference-probe-* and candidate-probe-* bind full vectors at --probe-steps.\nScore attachments and KL probes are supplied independently; neither supplies the other implicitly.\nEvery selected-step KL upper bound must meet the budget; an unresolved interval is unknown.\nUse run argument arrays accept these paired-observation options without --relation or --budget.\nA successful command reports accept, refute or unknown; it does not grant use admission.\nModel substitution remains unknown." options
