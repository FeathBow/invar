{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Stream (Sample (..), Signal (..), Algorithm (..), Plan (..), Current (..), Applied (..), Reply (..), ReferenceSource (..), Stream, Error (..), sourceName, readSource, engineWords, begin, proximal, reference, current, applied, complete, digest, observationOf, proximals, referencesOf, sourceOf, currents, identity, exchange, opening, completions) where

import Control.Monad (unless, when, zipWithM)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.ByteString qualified as Bytes
import Data.ByteString.Builder qualified as Builder
import Data.ByteString.Lazy qualified as Lazy
import Data.List (genericLength, zip5)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.Encoding qualified as Text
import Data.Word (Word32)
import GHC.Float (castDoubleToWord64)
import Invar.Artifact qualified as Artifact
import Invar.Async.Completion.Internal (Completion (Completion))
import Invar.Async.Completion.Internal qualified as Completion
import Invar.Learn.Algorithm qualified as Definition
import Invar.Learn.Objective qualified as Objective
import Invar.Spec.Invocation qualified as V
import Numeric.Natural (Natural)

data Sample = Sample {name :: Text, behaviorWords :: [Word32], referenceWords :: [Word32], signal :: Signal}
    deriving (Eq, Show)

data Signal = Advantage Word32 | Teacher [Word32]
    deriving (Eq, Show)

data Algorithm = Grouped | Distilled Definition.Teacher
    deriving (Eq, Show)

data Current = Current {step :: Natural, sample :: Text, words32 :: [Word32], observation :: String, state :: String}
    deriving (Eq, Show)

data Applied = Applied {appliedStep :: Natural, before :: String, after :: String, consumed :: [String]}
    deriving (Eq, Show)

data Reply = Reply {replyStep :: Natural, replySample :: Text, replyObservation :: String, replyState :: String, objective :: [Word32], surrogate :: [Word32]}
    deriving (Eq, Show)

data Error
    = UnknownSample Text
    | OutOfOrder Natural Text
    | StepMismatch Natural Natural
    | StateMismatch Natural String String
    | LengthMismatch Text
    | DuplicateProximal Text
    | LateProximal Text
    | MissingProximal Text
    | DuplicateReference Text
    | UnexpectedReference Text
    | MissingReference Text
    | Incomplete Natural
    | ConsumedMismatch Natural
    | ObservationMismatch Text
    | Finished
    | Unfinished
    | Scalar Objective.Error
    | InvalidPlan String
    deriving (Eq, Show)

data ReferenceSource = FromEngine | FromLearner
    deriving (Eq, Show)

data Plan = Plan {coefficients :: Objective.Profile, origin :: String, members :: [Sample], batches :: [[Text]], source :: ReferenceSource, algorithm :: Algorithm}
    deriving (Eq, Show)

sourceName :: ReferenceSource -> Text
sourceName FromEngine = "engine"
sourceName FromLearner = "learner"

readSource :: Text -> Maybe ReferenceSource
readSource name = lookup name [(sourceName chosen, chosen) | chosen <- [FromEngine, FromLearner]]

engineWords :: Sample -> [Word32]
engineWords entry = if null (referenceWords entry) then behaviorWords entry else referenceWords entry

data Stream = Stream
    { binding :: V.Binding
    , planned :: String
    , profile :: Objective.Profile
    , samples :: Map Text Sample
    , plan :: [[Text]]
    , position :: Natural
    , answered :: [String]
    , expected :: String
    , fixed :: Map Text [Word32]
    , observed :: [(Natural, Text, [Word32])]
    , closed :: [Completion]
    , referenceSource :: ReferenceSource
    , references :: Map Text [Word32]
    }
    deriving (Eq, Show)

begin :: V.Binding -> Plan -> Either Error Stream
begin bound (Plan chosen initial declared steps scoring kind) = do
    let named = Map.fromList [(name entry, entry) | entry <- declared]
    when (null declared || Map.size named /= length declared) (Left (InvalidPlan "Samples must be nonempty and distinct"))
    when (any (null . behaviorWords) declared) (Left (InvalidPlan "Every sample needs at least one response token"))
    when (any (\entry -> not (null (referenceWords entry)) && length (referenceWords entry) /= length (behaviorWords entry)) declared) (Left (InvalidPlan "Reference words must cover every response token"))
    unless (all (carries kind) declared) (Left (InvalidPlan "Every sample must carry the signal of the plan's algorithm, and teacher words must cover every response token"))
    when (null steps || any null steps) (Left (InvalidPlan "Optimizer steps must be nonempty"))
    unless (Set.fromList (concat steps) == Map.keysSet named) (Left (InvalidPlan "Optimizer steps must use every sample and no other"))
    pure (Stream bound (planIdentity chosen initial declared steps scoring kind) chosen named steps 0 [] initial Map.empty [] [] scoring Map.empty)

carries :: Algorithm -> Sample -> Bool
carries Grouped entry = case signal entry of
    Advantage _ -> True
    Teacher _ -> False
carries (Distilled _) entry = case signal entry of
    Teacher words32 -> length words32 == length (behaviorWords entry)
    Advantage _ -> False

planIdentity :: Objective.Profile -> String -> [Sample] -> [[Text]] -> ReferenceSource -> Algorithm -> String
planIdentity chosen initial declared steps scoring kind = Artifact.hex (SHA256.hash (Lazy.toStrict (Builder.toLazyByteString encoded)))
  where
    encoded = algorithmTag kind <> Builder.char7 (sourceTag scoring) <> Builder.word64LE (castDoubleToWord64 (Objective.epsilon chosen)) <> Builder.word64LE (castDoubleToWord64 (Objective.penalty chosen)) <> text (Text.pack initial) <> counted (map entry declared) <> counted (map (counted . map text) steps)
    algorithmTag Grouped = mempty
    algorithmTag (Distilled teacher) = Builder.char7 'd' <> foldMap (text . Text.pack . ($ teacher)) [Definition.teacherModel, Definition.teacherRevision, Definition.teacherAdapter, Definition.teacherTokenizer, Definition.teacherBase, Definition.teacherAssembly] <> Builder.word64LE (castDoubleToWord64 (Definition.teacherTemperature teacher)) <> foldMap (text . Text.pack . ($ teacher)) [Definition.teacherTruncation, Definition.teacherLogprobs, Definition.teacherPath]
    sourceTag FromEngine = 'e'
    sourceTag FromLearner = 'l'
    entry value = text (name value) <> counted (map Builder.word32LE (behaviorWords value)) <> counted (map Builder.word32LE (referenceWords value)) <> signalBytes (signal value)
    signalBytes (Advantage word) = Builder.word32LE word
    signalBytes (Teacher words32) = counted (map Builder.word32LE words32)
    text value = let bytes = Text.encodeUtf8 value in Builder.word64LE (fromIntegral (Bytes.length bytes)) <> Builder.byteString bytes
    counted values = Builder.word64LE (fromIntegral (length values)) <> mconcat values

proximal :: Stream -> Text -> [Word32] -> Either Error Stream
proximal stream named values = do
    entry <- lookupSample stream named
    when (position stream > 0 || not (null (answered stream))) (Left (LateProximal named))
    when (named `elem` firstStep stream) (Left (DuplicateProximal named))
    when (Map.member named (fixed stream)) (Left (DuplicateProximal named))
    unless (length values == length (behaviorWords entry)) (Left (LengthMismatch named))
    pure stream {fixed = Map.insert named values (fixed stream)}

reference :: Stream -> Text -> [Word32] -> Either Error Stream
reference stream named values = do
    when (referenceSource stream == FromEngine) (Left (UnexpectedReference named))
    entry <- lookupSample stream named
    when (Map.member named (references stream)) (Left (DuplicateReference named))
    unless (length values == length (behaviorWords entry)) (Left (LengthMismatch named))
    pure stream {references = Map.insert named values (references stream)}

current :: Stream -> Current -> Either Error (Stream, Reply)
current stream report = do
    batch <- maybe (Left Finished) Right (selected stream)
    let index = length (answered stream)
    next <- case drop index batch of
        chosen : _ -> Right chosen
        [] -> Left (OutOfOrder (step report) (sample report))
    unless (step report == position stream && sample report == next) (Left (OutOfOrder (step report) (sample report)))
    unless (state report == expected stream) (Left (StateMismatch (position stream) (expected stream) (state report)))
    entry <- lookupSample stream (sample report)
    unless (length (words32 report) == length (behaviorWords entry)) (Left (LengthMismatch (sample report)))
    unless (observation report == observationOf (words32 report)) (Left (ObservationMismatch (sample report)))
    let recorded = if position stream == 0 then Map.insert (sample report) (words32 report) (fixed stream) else fixed stream
    frozen <- maybe (Left (MissingProximal (sample report))) Right (Map.lookup (sample report) recorded)
    chosen <- case referenceSource stream of
        FromEngine -> pure (engineWords entry)
        FromLearner -> maybe (Left (MissingReference (sample report))) Right (Map.lookup (sample report) (references stream))
    advantages <- either (Left . Scalar) Right (advantagesOf entry frozen)
    let inputs = [Objective.Inputs {Objective.current = now, Objective.proximal = old, Objective.behavior = seen, Objective.fixed = ref, Objective.advantage = advantage} | (now, old, seen, ref, advantage) <- zip5 (words32 report) frozen (behaviorWords entry) chosen advantages]
    outputs <- either (Left . Scalar) Right (Objective.calculate (profile stream) (fromIntegral (denominator stream batch)) inputs)
    let reply = Reply (position stream) (sample report) (observation report) (state report) (map Objective.gradient outputs) (map Objective.surrogateGradient outputs)
    pure
        ( stream
            { answered = answered stream ++ [digest reply]
            , fixed = recorded
            , observed = observed stream ++ [(position stream, sample report, words32 report)]
            }
        , reply
        )

advantagesOf :: Sample -> [Word32] -> Either Objective.Error [Word32]
advantagesOf entry frozen = case signal entry of
    Advantage word -> Right (replicate (length frozen) word)
    Teacher words32 -> zipWithM Objective.distilled words32 frozen

applied :: Stream -> Applied -> Either Error (Stream, Completion)
applied stream report = do
    batch <- maybe (Left Finished) Right (selected stream)
    unless (appliedStep report == position stream) (Left (StepMismatch (position stream) (appliedStep report)))
    unless (length (answered stream) == length batch) (Left (Incomplete (position stream)))
    unless (before report == expected stream) (Left (StateMismatch (position stream) (expected stream) (before report)))
    unless (consumed report == answered stream) (Left (ConsumedMismatch (position stream)))
    when (position stream == 0) $ case [entry | entry <- Map.keys (samples stream), Map.notMember entry (fixed stream)] of
        missing : _ -> Left (MissingProximal missing)
        [] -> pure ()
    let done = Completion {Completion.binding = binding stream, Completion.plan = planned stream, Completion.step = position stream, Completion.consumed = consumedDigest (consumed report), Completion.before = before report, Completion.after = after report}
    pure (stream {position = position stream + 1, answered = [], expected = after report, closed = closed stream ++ [done]}, done)

complete :: Stream -> Either Error String
complete stream
    | position stream == genericLength (plan stream) = Right (expected stream)
    | otherwise = Left Unfinished

digest :: Reply -> String
digest reply = observationOf (objective reply ++ surrogate reply)

consumedDigest :: [String] -> String
consumedDigest values = Artifact.hex (SHA256.hash (Lazy.toStrict (Builder.toLazyByteString (foldMap (\value -> Builder.stringUtf8 value <> Builder.char7 '\n') values))))

observationOf :: [Word32] -> String
observationOf values = Artifact.hex (SHA256.hash (Lazy.toStrict (Builder.toLazyByteString (foldMap Builder.word32LE values))))

identity :: Stream -> String
identity = planned

exchange :: Stream -> V.Binding
exchange = binding

opening :: Stream -> String
opening = expected

completions :: Stream -> [Completion]
completions = closed

proximals :: Stream -> Map Text [Word32]
proximals = fixed

referencesOf :: Stream -> Map Text [Word32]
referencesOf = references

sourceOf :: Stream -> ReferenceSource
sourceOf = referenceSource

currents :: Stream -> [(Natural, Text, [Word32])]
currents = observed

selected :: Stream -> Maybe [Text]
selected stream = case drop (fromIntegral (position stream)) (plan stream) of
    batch : _ -> Just batch
    [] -> Nothing

firstStep :: Stream -> [Text]
firstStep stream = case plan stream of
    batch : _ -> batch
    [] -> []

denominator :: Stream -> [Text] -> Natural
denominator stream batch = sum [genericLength (behaviorWords entry) | named <- batch, Just entry <- [Map.lookup named (samples stream)]]

lookupSample :: Stream -> Text -> Either Error Sample
lookupSample stream named = maybe (Left (UnknownSample named)) Right (Map.lookup named (samples stream))
