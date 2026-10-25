{-# LANGUAGE OverloadedStrings #-}

module Invar.Remote.Control (
    Launching (..),
    Instance (..),
    Failure (..),
    Ending (..),
    Outcome (..),
    Signal (..),
    Signalled (..),
    Request (..),
    Refusal (..),
    Answer (..),
    Files (..),
    State (..),
    Observation (..),
    launchingFile,
    instanceFile,
    failureFile,
    outcomeFile,
    signalledFile,
    requestFile,
    refusalFile,
    admitRequest,
    answer,
    reply,
    answerName,
    observe,
    describe,
) where

import Control.Monad (unless, when)
import Data.Aeson (Key, Object, Value (..), encode, object, parseJSON, withObject, (.:), (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Pair, Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Lazy qualified as Lazy
import Data.Either (isLeft, isRight)
import Data.Maybe (isJust, isNothing)
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.Encoding (decodeUtf8Lenient, encodeUtf8)
import Invar.Json qualified as Json
import Numeric.Natural (Natural)

data Launching = Launching {host :: Text, nonce :: Text, wrapper :: Natural}
    deriving (Eq, Show)

data Instance = Instance {launching :: Launching, worker :: Natural}
    deriving (Eq, Show)

data Failure = Failure {failed :: Launching, reason :: Text}
    deriving (Eq, Show)

data Ending = Exited Int | Signalled Int
    deriving (Eq, Show)

data Outcome = Outcome {ended :: Instance, ending :: Ending, endedAt :: Integer, bytes :: Natural, digest :: String}
    deriving (Eq, Show)

data Signal = Terminate | Kill
    deriving (Eq, Show)

data Signalled = Sent {target :: Instance, signal :: Signal, sentAt :: Integer}
    deriving (Eq, Show)

data Request = Request {requested :: Text, requestedWrapper :: Natural, requestedWorker :: Natural, grace :: Natural}
    deriving (Eq, Show)

data Refusal = Refusal {held :: Instance, refusedRequest :: Text, refusedBecause :: Text, refusedAt :: Integer}
    deriving (Eq, Show)

data Answer = Unrequested | Unanswered | Confirmed | Refused
    deriving (Eq, Show)

data Files = Files {launchingBytes, instanceBytes, failureBytes, outcomeBytes, requestBytes, terminatedBytes, killedBytes :: Maybe ByteString, refusedBytes :: [ByteString]}

data State = Absent | Launched Launching | Started Instance | LaunchFailed (Maybe Instance) Failure | Ended Outcome
    deriving (Eq, Show)

data Observation = Observation {state :: State, asked :: Maybe ByteString, terminated :: Maybe Signalled, killed :: Maybe Signalled, refused :: [Refusal], alive :: Bool}
    deriving (Eq, Show)

launchingFile :: Launching -> ByteString
launchingFile = file . launchingValue

instanceFile :: Instance -> ByteString
instanceFile = file . instanceValue

failureFile :: Failure -> ByteString
failureFile = file . failureValue

outcomeFile :: Outcome -> ByteString
outcomeFile = file . outcomeValue

signalledFile :: Signalled -> ByteString
signalledFile = file . signalledValue

requestFile :: Request -> ByteString
requestFile (Request chosen wrapperId workerId seconds) = file (object ["format" .= ("invar-remote-request-v1" :: Text), "nonce" .= chosen, "wrapper" .= wrapperId, "worker" .= workerId, "grace" .= seconds])

file :: Value -> ByteString
file = Lazy.toStrict . encode

refusalFile :: Refusal -> ByteString
refusalFile = file . refusalValue

-- | The request in `terminate.json` is admitted when it names this instance. The wrapper signals for an admitted
-- request, which then stays in place; it records a refused one with its own refusal and removes it, so that a request
-- naming another instance does not close the termination of this one.
admitRequest :: Instance -> ByteString -> Either Text Request
admitRequest chosen encoded = case decoded "invar-remote-request-v1" ["nonce", "wrapper", "worker", "grace"] parser encoded of
    Left problem -> Left (Text.pack ("The termination request cannot be read: " ++ problem))
    Right value
        | requested value == nonce (launching chosen) && requestedWrapper value == wrapper (launching chosen) && requestedWorker value == worker chosen -> Right value
        | otherwise -> Left "The termination request names another instance"
  where
    parser fields = Request <$> fields .: "nonce" <*> fields .: "wrapper" <*> fields .: "worker" <*> fields .: "grace"

-- | The answer to the request the instance holds.
answer :: Observation -> Answer
answer current = case asked current of
    Nothing -> Unrequested
    Just encoded
        | isJust (terminated current) -> Confirmed
        | refusedOf encoded current -> Refused
        | otherwise -> Unanswered

-- | The answer to the request with these bytes: refused when the wrapper refused them, the answer to the request the
-- instance holds when it holds these bytes, and otherwise unrequested, so a request never takes another's answer.
reply :: ByteString -> Observation -> Answer
reply encoded current
    | refusedOf encoded current = Refused
    | asked current == Just encoded = answer current
    | otherwise = Unrequested

refusedOf :: ByteString -> Observation -> Bool
refusedOf encoded current = any ((== decodeUtf8Lenient encoded) . refusedRequest) (refused current)

answerName :: Answer -> Text
answerName value = case value of
    Unrequested -> "unrequested"
    Unanswered -> "unanswered"
    Confirmed -> "confirmed"
    Refused -> "refused"

-- | Every control file, decoded and checked against each other. The files appear in a fixed order, and only a refused
-- request is removed, after its refusal, so a later file names exactly the identity of the earlier ones.
observe :: Bool -> Files -> Either String Observation
observe holding supplied = do
    started <- traverse (decoded "invar-remote-launching-v1" launchingFields launchingParser) (launchingBytes supplied)
    published <- traverse (decoded "invar-remote-instance-v1" instanceFields instanceParser) (instanceBytes supplied)
    failure <- traverse (decoded "invar-remote-failure-v1" failureFields failureParser) (failureBytes supplied)
    outcome <- traverse (decoded "invar-remote-outcome-v1" outcomeFields outcomeParser) (outcomeBytes supplied)
    termination <- traverse (decoded "invar-remote-signalled-v1" signalledFields signalledParser) (terminatedBytes supplied)
    escalation <- traverse (decoded "invar-remote-signalled-v1" signalledFields signalledParser) (killedBytes supplied)
    refusals <- traverse (decoded "invar-remote-refused-v1" refusedFields refusalParser) (refusedBytes supplied)
    when (isJust failure && isJust outcome) (Left "A remote instance records both a launch failure and an outcome")
    current <- case started of
        Nothing -> do
            unless (isNothing published && isNothing failure && isNothing outcome && isNothing (requestBytes supplied) && isNothing termination && isNothing escalation && null refusals) (Left "Remote control files exist without a launch record")
            pure Absent
        Just first -> do
            mapM_ (\value -> unless (launching value == first) (Left "The remote instance names another launch")) published
            mapM_ (\value -> unless (failed value == first) (Left "The remote launch failure names another launch")) failure
            case (published, failure, outcome) of
                (_, Just value, _) -> pure (LaunchFailed published value)
                (Just value, _, Just result) -> unless (ended result == value) (Left "The remote outcome names another instance") >> pure (Ended result)
                (Nothing, _, Just _) -> Left "A remote outcome exists without its instance"
                (Just value, _, Nothing) -> pure (Started value)
                (Nothing, _, Nothing) -> pure (Launched first)
    let named = case current of
            Started value -> Just value
            Ended result -> Just (ended result)
            LaunchFailed value _ -> value
            _ -> Nothing
    signalled Terminate named termination
    signalled Kill named escalation
    when (isJust escalation && isNothing termination) (Left "A remote kill was recorded without its termination")
    mapM_ (\_ -> answered named (requestBytes supplied) isRight) termination
    mapM_ (\value -> unless (Just (held value) == named) (Left "A remote refusal names another instance") >> answered named (Just (encodeUtf8 (refusedRequest value))) isLeft) refusals
    pure (Observation current (requestBytes supplied) termination escalation refusals holding)
  where
    signalled chosen named = mapM_ $ \value -> do
        unless (Just (target value) == named) (Left "A remote signal record names another instance")
        unless (signal value == chosen) (Left "A remote signal record holds another signal")
    answered named request agrees = case (named, request) of
        (Just chosen, Just encoded) -> unless (agrees (admitRequest chosen encoded)) (Left "A remote answer disagrees with its termination request")
        _ -> Left "A remote answer exists without its instance and termination request"

describe :: Observation -> Value
describe observation@(Observation current pending termination escalation refusals holding) =
    object
        [ "format" .= ("invar-remote-status-v1" :: Text)
        , "state" .= stateName current
        , "launching" .= fmap launchingValue (launched current)
        , "instance" .= fmap instanceValue (instanceOf current)
        , "failure" .= fmap failureValue (failureOf current)
        , "outcome" .= fmap outcomeValue (outcomeOf current)
        , "request" .= fmap decodeUtf8Lenient pending
        , "terminated" .= fmap signalledValue termination
        , "killed" .= fmap signalledValue escalation
        , "refused" .= map refusalValue refusals
        , "answer" .= answerName (answer observation)
        , "wrapper_alive" .= holding
        ]
  where
    stateName value = case value of
        Absent -> "absent" :: Text
        Launched _ -> "launching"
        Started _ -> "started"
        LaunchFailed _ _ -> "launch_failed"
        Ended _ -> "ended"
    launched value = case value of
        Absent -> Nothing
        Launched chosen -> Just chosen
        Started chosen -> Just (launching chosen)
        LaunchFailed _ chosen -> Just (failed chosen)
        Ended chosen -> Just (launching (ended chosen))
    instanceOf value = case value of
        Started chosen -> Just chosen
        LaunchFailed chosen _ -> chosen
        Ended chosen -> Just (ended chosen)
        _ -> Nothing
    failureOf value = case value of
        LaunchFailed _ chosen -> Just chosen
        _ -> Nothing
    outcomeOf value = case value of
        Ended chosen -> Just chosen
        _ -> Nothing

decoded :: Text -> [Key] -> (Object -> Parser value) -> ByteString -> Either String value
decoded format names parser encoded = Json.decode encoded >>= parseEither (withObject "remote control record" checked)
  where
    checked fields = do
        Json.fields ("format" : names) fields
        declared <- fields .: "format"
        unless (declared == format) (fail ("Expected " <> show format))
        parser fields

launchingFields, instanceFields, failureFields, outcomeFields, signalledFields, refusedFields :: [Key]
launchingFields = ["host", "nonce", "wrapper"]
instanceFields = launchingFields ++ ["worker", "group"]
failureFields = launchingFields ++ ["reason"]
outcomeFields = instanceFields ++ ["ending", "ended", "bytes", "sha256"]
signalledFields = instanceFields ++ ["signal", "at"]
refusedFields = instanceFields ++ ["request", "reason", "at"]

launchingParser :: Object -> Parser Launching
launchingParser fields = Launching <$> fields .: "host" <*> fields .: "nonce" <*> fields .: "wrapper"

instanceParser :: Object -> Parser Instance
instanceParser fields = do
    chosen <- launchingParser fields
    workerId <- fields .: "worker"
    group <- fields .: "group"
    unless (group == workerId) (fail "The worker must lead its own process group")
    pure (Instance chosen workerId)

failureParser :: Object -> Parser Failure
failureParser fields = Failure <$> launchingParser fields <*> fields .: "reason"

outcomeParser :: Object -> Parser Outcome
outcomeParser fields = do
    chosen <- instanceParser fields
    result <- fields .: "ending" >>= withObject "remote ending" endingParser
    sha <- fields .: "sha256" >>= Json.identity
    Outcome chosen result <$> fields .: "ended" <*> fields .: "bytes" <*> pure sha
  where
    endingParser value = case Fields.toList value of
        [("exited", code)] -> Exited <$> parseJSON code
        [("signalled", number)] -> Signalled <$> parseJSON number
        _ -> fail "A remote ending is either exited or signalled"

signalledParser :: Object -> Parser Signalled
signalledParser fields = do
    chosen <- instanceParser fields
    named <- fields .: "signal"
    sent <- case named :: Text of
        "TERM" -> pure Terminate
        "KILL" -> pure Kill
        _ -> fail "Unknown remote signal"
    Sent chosen sent <$> fields .: "at"

refusalParser :: Object -> Parser Refusal
refusalParser fields = Refusal <$> instanceParser fields <*> fields .: "request" <*> fields .: "reason" <*> fields .: "at"

refusalValue :: Refusal -> Value
refusalValue (Refusal chosen asking why at) = object (instancePairs chosen ++ ["format" .= ("invar-remote-refused-v1" :: Text), "request" .= asking, "reason" .= why, "at" .= at])

launchingValue :: Launching -> Value
launchingValue chosen = object (launchingPairs chosen ++ ["format" .= ("invar-remote-launching-v1" :: Text)])

launchingPairs :: Launching -> [Pair]
launchingPairs (Launching name chosen wrapperId) = ["host" .= name, "nonce" .= chosen, "wrapper" .= wrapperId]

instanceValue :: Instance -> Value
instanceValue chosen = object (instancePairs chosen ++ ["format" .= ("invar-remote-instance-v1" :: Text)])

instancePairs :: Instance -> [Pair]
instancePairs (Instance chosen workerId) = launchingPairs chosen ++ ["worker" .= workerId, "group" .= workerId]

failureValue :: Failure -> Value
failureValue (Failure chosen why) = object (launchingPairs chosen ++ ["format" .= ("invar-remote-failure-v1" :: Text), "reason" .= why])

outcomeValue :: Outcome -> Value
outcomeValue (Outcome chosen result at count sha) =
    object (instancePairs chosen ++ ["format" .= ("invar-remote-outcome-v1" :: Text), "ending" .= endingValue result, "ended" .= at, "bytes" .= count, "sha256" .= sha])
  where
    endingValue (Exited code) = object ["exited" .= code]
    endingValue (Signalled number) = object ["signalled" .= number]

signalledValue :: Signalled -> Value
signalledValue (Sent chosen sent at) = object (instancePairs chosen ++ ["format" .= ("invar-remote-signalled-v1" :: Text), "signal" .= signalName sent, "at" .= at])
  where
    signalName Terminate = "TERM" :: Text
    signalName Kill = "KILL"
