module Invar.Learn.Stream (Sample (..), Current (..), Applied (..), Reply (..), Stream, Error (..), begin, proximal, current, applied, complete, digest, observationOf, proximals, currents, losses) where

import Control.Monad (unless, when)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.ByteString.Builder qualified as Builder
import Data.ByteString.Lazy qualified as Lazy
import Data.List (genericLength, zip4)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Word (Word32)
import Invar.Artifact qualified as Artifact
import Invar.Learn.Objective qualified as Objective
import Numeric.Natural (Natural)

data Sample = Sample {name :: Text, behaviorWords :: [Word32], referenceWords :: [Word32], advantageWord :: Word32}
    deriving (Eq, Show)

data Current = Current {step :: Natural, sample :: Text, words32 :: [Word32], observation :: String, state :: String}
    deriving (Eq, Show)

data Applied = Applied {appliedStep :: Natural, before :: String, after :: String, consumed :: [String]}
    deriving (Eq, Show)

data Reply = Reply {replyStep :: Natural, replySample :: Text, replyObservation :: String, replyState :: String, objective :: [Word32], reward :: [Word32]}
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
    | Incomplete Natural
    | ConsumedMismatch Natural
    | ObservationMismatch Text
    | Finished
    | Unfinished
    | Scalar Objective.Error
    deriving (Eq, Show)

data Stream = Stream
    { profile :: Objective.Profile
    , samples :: Map Text Sample
    , plan :: [[Text]]
    , position :: Natural
    , answered :: [String]
    , expected :: String
    , fixed :: Map Text [Word32]
    , observed :: [(Natural, Text, [Word32])]
    , terms :: Map Natural [Word32]
    }
    deriving (Eq, Show)

begin :: Objective.Profile -> String -> [Sample] -> [[Text]] -> Stream
begin chosen initial declared steps = Stream chosen (Map.fromList [(name entry, entry) | entry <- declared]) steps 0 [] initial Map.empty [] Map.empty

proximal :: Stream -> Text -> [Word32] -> Either Error Stream
proximal stream named values = do
    entry <- lookupSample stream named
    when (position stream > 0 || not (null (answered stream))) (Left (LateProximal named))
    when (named `elem` firstStep stream) (Left (DuplicateProximal named))
    when (Map.member named (fixed stream)) (Left (DuplicateProximal named))
    unless (length values == length (behaviorWords entry)) (Left (LengthMismatch named))
    pure stream {fixed = Map.insert named values (fixed stream)}

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
    let references = if null (referenceWords entry) then behaviorWords entry else referenceWords entry
        inputs = [Objective.Inputs {Objective.current = now, Objective.proximal = old, Objective.behavior = seen, Objective.fixed = ref, Objective.advantage = advantageWord entry} | (now, old, seen, ref) <- zip4 (words32 report) frozen (behaviorWords entry) references]
    outputs <- either (Left . Scalar) Right (Objective.calculate (profile stream) (fromIntegral (denominator stream batch)) inputs)
    let reply = Reply (position stream) (sample report) (observation report) (state report) (map Objective.gradient outputs) (map Objective.rewardGradient outputs)
    pure
        ( stream
            { answered = answered stream ++ [digest reply]
            , fixed = recorded
            , observed = observed stream ++ [(position stream, sample report, words32 report)]
            , terms = Map.insertWith (flip (++)) (position stream) (map Objective.term outputs) (terms stream)
            }
        , reply
        )

applied :: Stream -> Applied -> Either Error Stream
applied stream report = do
    batch <- maybe (Left Finished) Right (selected stream)
    unless (appliedStep report == position stream) (Left (StepMismatch (position stream) (appliedStep report)))
    unless (length (answered stream) == length batch) (Left (Incomplete (position stream)))
    unless (before report == expected stream) (Left (StateMismatch (position stream) (expected stream) (before report)))
    unless (consumed report == answered stream) (Left (ConsumedMismatch (position stream)))
    when (position stream == 0) $ case [entry | entry <- Map.keys (samples stream), Map.notMember entry (fixed stream)] of
        missing : _ -> Left (MissingProximal missing)
        [] -> pure ()
    pure stream {position = position stream + 1, answered = [], expected = after report}

complete :: Stream -> Either Error String
complete stream
    | position stream == genericLength (plan stream) = Right (expected stream)
    | otherwise = Left Unfinished

digest :: Reply -> String
digest reply = observationOf (objective reply ++ reward reply)

observationOf :: [Word32] -> String
observationOf values = Artifact.hex (SHA256.hash (Lazy.toStrict (Builder.toLazyByteString (foldMap Builder.word32LE values))))

proximals :: Stream -> Map Text [Word32]
proximals = fixed

currents :: Stream -> [(Natural, Text, [Word32])]
currents = observed

losses :: Stream -> Either Error [Word32]
losses stream = traverse (either (Left . Scalar) Right . Objective.mean32) (Map.elems (terms stream))

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
