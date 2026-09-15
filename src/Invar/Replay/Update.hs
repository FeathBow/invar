{-# LANGUAGE OverloadedStrings #-}

module Invar.Replay.Update (Run (..), Mode (..), Reference, Update, admit, admitResident, admitShared, describe, updates, value, decode, decodeBytes, decodeMany, successors, report, consumed, checkpoint, published) where

import Control.Monad (foldM, unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Aeson (FromJSON (parseJSON), Object, Value (..), object, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Parser, parseEither)
import Data.Bifunctor (first)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.List (nub)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8, encodeUtf8)
import Invar.Artifact qualified as Artifact
import Invar.Infer.Framing qualified as Framing
import Invar.Infer.Wire qualified as Wire
import Invar.Json qualified as Json
import Invar.Learn.Protocol qualified as Protocol
import Invar.Learn.Report qualified as Report
import Invar.Measurement.Duration qualified as Duration
import Invar.Resident qualified as Boundary
import Invar.Resident.Observation qualified as Resident
import Invar.Spec.Invocation qualified as V
import Numeric.Natural (Natural)

data Run = Run {initial :: FilePath, count :: Natural, exitCode :: Int}
data Mode = Finite | Resident | Shared deriving (Eq, Show)
data Reference = Reference String Text [Update] (Maybe (Mode, [Resident.Group], Duration.Duration))
data Update = Update {report :: Report.Report, consumed :: Object, consumedBytes :: ByteString, checkpoint :: FilePath, published :: FilePath}
data Frame = Frame {position :: Natural, raw :: ByteString, fields :: Object}

-- Directory observations are supplied by the filesystem boundary. This report
-- describes replay inputs; it is not a training-history or publication receipt.
admit :: (FilePath -> IO Bool) -> Run -> ByteString -> IO Reference
admit = admitWith Finite

admitResident :: (FilePath -> IO Bool) -> Run -> ByteString -> IO Reference
admitResident = admitWith Resident

admitShared :: (FilePath -> IO Bool) -> Run -> ByteString -> IO Reference
admitShared = admitWith Shared

admitWith :: Mode -> (FilePath -> IO Bool) -> Run -> ByteString -> IO Reference
admitWith resident directory run encoded = do
    observed <- either invalid pure (reference resident run encoded)
    mapM_ (\update -> directory (checkpoint update) >>= \exists -> unless exists (invalid "Update input checkpoint is not a directory")) (updates observed)
    pure observed

reference :: Mode -> Run -> ByteString -> Either String Reference
reference mode run encoded = do
    unless (exitCode run == 0) (Left "Training process did not exit successfully")
    unless (count run > 0) (Left "Expected a positive update count")
    _ <- parseEither path (String (Text.pack (initial run)))
    unless ("\n" `Bytes.isSuffixOf` encoded) (Left "Incomplete training stream")
    frames <- traverse frame (zip [0 ..] (Bytes.lines encoded))
    let requests = filter updateInput frames
        results = filter (\event -> stage "result" event && Fields.member "update" (fields event)) frames
        publications = filter (phase "published") frames
        cycles = filter (phase "cycle") frames
        expected = fromIntegral (count run)
    unless (all ((== expected) . length) [requests, results, publications]) (Left "Training stream does not contain exactly the expected consumed, result and published records")
    let resident = mode /= Finite
        (closing, preceding) = if resident then span (stage "closed") (reverse frames) else ([], reverse frames)
    when resident (unless (length closing == length (filter (stage "closed") frames)) (Left "Resident process close precedes the terminal training boundary"))
    ending <- terminal (reverse preceding) publications cycles
    indexed <- invocations frames
    paired <- traverse (associate indexed) (zip requests publications)
    let bindings = map (\(_, _, _, binding, _) -> binding) paired
    unless (distinct (map V.boundCall bindings) && distinct (map V.boundAttempt bindings) && distinct (map V.boundInstance bindings)) (Left "Repeated update invocation identity")
    let locations = map (\(_, _, _, _, destination) -> destination) paired
        inputs = initial run : locations
    mapM_ boundary (zip publications (drop 1 requests))
    mapM_ boundary (zip cycles (drop 1 requests))
    let planned = [Update observed consumption original input destination | (input, (observed, consumption, original, _, destination)) <- zip inputs paired]
    successors planned
    lifetime <- case mode of
        Finite -> pure Nothing
        Resident -> do
            (groups, elapsed) <- residentLearning planned frames (reverse closing)
            pure (Just (mode, groups, elapsed))
        Shared -> do
            (groups, elapsed) <- sharedLearning planned frames
            pure (Just (mode, groups, elapsed))
    pure (Reference (Artifact.hex (SHA256.hash encoded)) ending planned lifetime)
  where
    distinct values = Set.size (Set.fromList values) == length values
    boundary (publishedEvent, next) = unless (position publishedEvent < position next) (Left "Next update consumption precedes the previous publication")

invocations :: [Frame] -> Either String (Map V.CallId [Frame])
invocations frames = do
    selected <- traverse bound [event | event <- frames, stage "consumed" event || stage "result" event, Fields.member "binding" (fields event)]
    pure (Map.map reverse (Map.fromListWith (++) selected))
  where
    bound event = do
        binding <- parseEither Wire.binding (fields event)
        pure (V.boundCall binding, [event])

residentLearning :: [Update] -> [Frame] -> [Frame] -> Either String ([Resident.Group], Duration.Duration)
residentLearning planned frames closing = do
    unless (length (filter (stage "loaded_learner") frames) == length planned) (Left "Resident reference learner inventory differs from its updates")
    let owner = Boundary.Owner Boundary.Learning 0
    (current, groups, _) <- foldM step (Resident.empty owner, [], map encoded frames) planned
    (_, elapsed, rest) <- Resident.finish current (map encoded closing)
    mapM_ (inferenceClose frames) rest
    owners <- traverse (parseEither (.: "owner") . Framing.fields) rest :: Either String [Value]
    unless (length owners == length (nub owners)) (Left "Repeated resident inference close")
    let learningOwner = object ["role" .= ("learning" :: Text), "session" .= (0 :: Natural)]
    unless (all (\record -> Fields.lookup "owner" (fields record) `elem` map Just (learningOwner : owners)) (filter (stage "released") frames)) (Left "Resident reference release has no final process close")
    let releases = [event | event <- frames, stage "released" event, Fields.lookup "owner" (fields event) == Just (object ["role" .= ("learning" :: Text), "session" .= (0 :: Natural)])]
    unless (length releases == length planned) (Left "Resident reference learner release inventory differs from its updates")
    pure (reverse groups, elapsed)
  where
    encoded event = Framing.Frame (raw event) (fields event)
    step (current, groups, remaining) update = do
        let (before, loaded) = break ((== Just (String "loaded_learner")) . Fields.lookup "stage" . Framing.fields) remaining
            prefix = reverse (takeWhile preparation (reverse before))
        (next, group, rest) <- Resident.learningObserved current (report update) (prefix ++ loaded)
        let actual = [Framing.fields event | event <- Resident.body group, Fields.lookup "stage" (Framing.fields event) == Just (String "consumed")]
        unless (actual == [consumed update]) (Left "Resident reference consumed a different update")
        pure (next, group : groups, rest)
    preparation event = Fields.lookup "stage" (Framing.fields event) `elem` map (Just . String) ["loading", "profile", "load", "activation"]

sharedLearning :: [Update] -> [Frame] -> Either String ([Resident.Group], Duration.Duration)
sharedLearning planned frames = do
    let owner = Boundary.Owner Boundary.Shared 0
        source = [Framing.Frame (raw event) (fields event) | event <- frames]
    (current, groups, remaining) <- foldM step (Resident.empty owner, [], source) (zip [0 ..] planned)
    (_, elapsed, rest) <- Resident.finish current remaining
    unless (null rest) (Left "Output follows the shared reference process close")
    pure (reverse groups, elapsed)
  where
    step (current, groups, remaining) (cohort, update) = do
        (active, inference, learning) <- Resident.inference (cohort, current) remaining
        (next, learned, rest) <- Resident.learningObserved active (report update) learning
        let actual = [Framing.fields event | event <- Resident.body learned, Fields.lookup "stage" (Framing.fields event) == Just (String "consumed")]
        unless (actual == [consumed update]) (Left "Shared reference consumed a different update")
        following <- case rest of
            publication : following | Fields.lookup "phase" (Framing.fields publication) == Just (String "published") -> pure following
            _ -> Left "Shared update release has no following publication"
        let afterCycle = case following of
                completed : after | Fields.lookup "phase" (Framing.fields completed) == Just (String "cycle") -> after
                _ -> following
        pure (next, learned : inference : groups, afterCycle)

-- Inference remains outside update replay's authority. Its trailing close must
-- nevertheless account for its original release inventory before process EOF.
inferenceClose :: [Frame] -> Framing.Frame -> Either String ()
inferenceClose frames event = do
    (role, session, groups) <- parseEither parse (Framing.fields event)
    unless (role == ("inference" :: Text)) (Left "Unexpected role after resident learner close")
    let owner = object ["role" .= role, "session" .= session]
        released = [() | record <- frames, stage "released" record, Fields.lookup "owner" (fields record) == Just owner]
    unless (groups == fromIntegral (length released)) (Left "Resident inference close differs from its original releases")
    _ <- Boundary.closed (Boundary.Owner Boundary.Inference session) groups (Framing.raw event)
    pure ()
  where
    parse fields_ = do
        owner <- fields_ .: "owner"
        (,,) <$> owner .: "role" <*> owner .: "session" <*> fields_ .: "groups"

successors :: [Update] -> Either String ()
successors planned = mapM_ successor (zip planned (drop 1 planned))

successor :: (Update, Update) -> Either String ()
successor (previous, next) = do
    policy <- Report.artifact "adapter" (report previous)
    learner <- Report.artifact "learner" (report previous)
    actual <- parseEither (withObject "successor update request" (\input -> (,) <$> input .: "policy" <*> input .: "learner")) (Report.request (report next))
    unless (actual == (policy, learner)) (Left "Successor update input differs from the previous publication")

frame :: (Natural, ByteString) -> Either String Frame
frame (index, encoded) = Frame index encoded <$> (Json.decode encoded >>= parseEither (withObject "training replay record" pure))

stage :: Text -> Frame -> Bool
stage name event = Fields.lookup "stage" (fields event) == Just (String name)

phase :: Text -> Frame -> Bool
phase name event = Fields.lookup "phase" (fields event) == Just (String name)

updateInput :: Frame -> Bool
updateInput event =
    stage "consumed" event && case Fields.lookup "request" (fields event) of
        Just (Object requested) -> Fields.member "samples" requested
        _ -> False

terminal :: [Frame] -> [Frame] -> [Frame] -> Either String Text
terminal frames publications [] = do
    unless (not (null frames) && map position (take 1 (reverse frames)) == map position (take 1 (reverse publications))) (Left "Training stream does not end with its final publication")
    pure "publication count and exit status"
terminal frames publications cycles = do
    unless (length cycles == length publications) (Left "Training stream cycle records differ from the expected updates")
    mapM_ check (zip [0 ..] (zip publications cycles))
    unless (map position (take 1 (reverse frames)) == map position (take 1 (reverse cycles))) (Left "Training stream does not end with its final cycle record")
    mapM_ (\(completed, next) -> unless (position completed < position next) (Left "Training cycle follows the next publication")) (zip cycles (drop 1 publications))
    pure "cycle records"
  where
    check (index, (publication, completed)) = do
        actual <- parseEither (.: "index") (fields completed) :: Either String Natural
        unless (actual == index && position publication < position completed) (Left "Training cycle index or publication order differs")

associate :: Map V.CallId [Frame] -> (Frame, Frame) -> Either String (Report.Report, Object, ByteString, V.Binding, FilePath)
associate indexed (input, publication) = do
    bound <- parseEither Wire.binding (fields input)
    let V.CallId call = V.boundCall bound
    selected <- maybe (Left "Missing update invocation") pure (Map.lookup (V.boundCall bound) indexed)
    observed <- Report.admit call (Bytes.unlines (map raw selected))
    validatePair observed (fields input)
    destination <- parseEither (publicationPath observed) (fields publication)
    result <- case selected of
        [_, event] | stage "result" event -> pure event
        _ -> Left "Expected one update result for a consumed request"
    unless (position input < position result && position result < position publication) (Left "Update consumption, result and publication are out of order")
    pure (observed, fields input, raw input, bound, destination)

validatePair :: Report.Report -> Object -> Either String ()
validatePair observed input = do
    parseEither (Json.fields ["stage", "binding", "program", "request", "load"]) input
    load <- parseEither (.: "load") input
    parseEither (Json.fields ["binding", "program"]) load
    binding <- parseEither Wire.binding load
    expected <- parseEither Wire.binding input
    program <- parseEither (.: "program") load :: Either String Text
    unless (binding == expected && not (Text.null program)) (Left "Update load invocation differs from consumption")
    output <- parseEither (withObject "update result" pure) (Report.result observed)
    parseEither (Json.fields ["stage", "binding", "request", "update", "gradients", "probabilities", "adapter", "learner", "storage"]) output
    first show (Protocol.validateSummary (Report.request observed) output)

publicationPath :: Report.Report -> Object -> Parser FilePath
publicationPath observed declared = do
    expected <- withObject "update invocation" (.: "binding") (Report.invocation observed)
    actual <- declared .: "binding"
    policy <- declared .: "policy" >>= Json.identity
    learner <- declared .: "learner" >>= Json.identity
    expectedPolicy <- either fail pure (Report.artifact "adapter" observed)
    expectedLearner <- either fail pure (Report.artifact "learner" observed)
    unless (actual == (expected :: Value) && policy == expectedPolicy && learner == expectedLearner) (fail "Published binding or identities differ from the update result")
    declared .: "checkpoint" >>= path

path :: Value -> Parser FilePath
path value_ = do
    decoded <- case value_ of String text -> pure (Text.unpack text); _ -> fail "Expected a checkpoint path"
    when (null decoded || '\0' `elem` decoded) (fail "Expected a nonempty checkpoint path without NUL")
    pure decoded

updates :: Reference -> [Update]
updates (Reference _ _ observed _) = observed

describe :: Reference -> Value
describe (Reference digest ending observed lifetime) = object (["reference_log_sha256" .= digest, "terminal" .= ending, "updates" .= map value observed] ++ resident)
  where
    resident = case lifetime of
        Nothing -> []
        Just (mode, groups, elapsed) -> ["mode" .= (if mode == Shared then "shared" else "resident" :: Text), "owner" .= (0 :: Natural), "groups" .= map Resident.describe groups, "close" .= elapsed]

value :: Update -> Value
value update = object ["consumed_json" .= decodeUtf8 (consumedBytes update), "result_json" .= decodeUtf8 (Report.output (report update)), "checkpoint" .= checkpoint update, "published" .= published update]

decode :: Value -> Either String Update
decode encoded = do
    (input, output, previous, destination) <- parseEither (withObject "update replay input" parse) encoded
    event <- frame (0, encodeUtf8 input)
    binding <- parseEither Wire.binding (fields event)
    let V.CallId call = V.boundCall binding
    observed <- Report.admit call (Bytes.unlines [encodeUtf8 input, encodeUtf8 output])
    validatePair observed (fields event)
    pure (Update observed (fields event) (encodeUtf8 input) previous destination)
  where
    parse object_ = do
        Json.fields ["consumed_json", "result_json", "checkpoint", "published"] object_
        (,,,) <$> object_ .: "consumed_json" <*> object_ .: "result_json" <*> (object_ .: "checkpoint" >>= path) <*> (object_ .: "published" >>= path)

decodeBytes :: ByteString -> Either String Update
decodeBytes encoded = Json.decode encoded >>= decode

decodeMany :: ByteString -> Either String [Update]
decodeMany encoded = (Json.decode encoded >>= parseEither parseJSON) >>= traverse decode

invalid :: String -> IO value
invalid = ioError . userError
