{-# LANGUAGE OverloadedStrings #-}

module ResidentFixture (Exchange (..), Scenario (..), owner, adapter, timer, prepare, prepareWith, scenario, scenarioWith, script, scriptWith, run, concluded, interrupted) where

import BatchCalls qualified as Serial
import BatchedProtocol qualified as Batch
import Calls qualified as Fixture
import Data.Aeson (Value (..), object, withObject, (.:), (.=))
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.IORef (modifyIORef', newIORef, readIORef, writeIORef)
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8)
import Hedgehog
import Invar.Artifact qualified as Artifact
import Invar.Infer.Invocation qualified as Call
import Invar.Infer.Trajectory (Trajectory)
import Invar.Transcript qualified as Transcript
import Invar.Worker qualified as Worker
import Invar.Worker.Resident qualified as Resident
import Numeric.Natural (Natural)
import System.FilePath ((</>))
import System.IO.Error (ioeGetErrorString, tryIOError)

data Exchange = Exchange {calls :: [Call.Call], before :: [Value], after :: [Value], permission :: ByteString, release :: Value, released :: Value}
data Scenario = Scenario {groups :: [Exchange], closed :: Value, ending :: String}

owner :: Natural
owner = 2

adapter :: FilePath
adapter = "resident adapter path"

format :: Text
format = "invar-resident-v1"

identity :: Natural -> Value
identity selected = object ["role" .= ("inference" :: Text), "session" .= selected]

timer :: Text -> Value
timer stage = object ["stage" .= stage, "cpu_seconds" .= (0.25 :: Double)]

measurement :: Text -> Text
measurement = decodeUtf8 . Fixture.wire . pure . timer

prepare :: FilePath -> Natural -> [Natural] -> PropertyT IO Exchange
prepare root index bindings = do
    requests <- Batch.setup bindings
    prepareWith root (owner, index) requests

prepareWith :: FilePath -> (Natural, Natural) -> [(Call.Call, [Value])] -> PropertyT IO Exchange
prepareWith root (selected, index) requests = do
    let prefix = if index == 0 then Batch.prefix requests else timer "activation" : drop 1 (Batch.prefix requests)
        suffix = Batch.suffix requests
        path = root </> ("group" ++ show index ++ ".jsonl")
    loads <-
        traverse
            ( \(_, values) -> case values of
                loaded : _ -> evalEither (parseEither (withObject "fixture load" (.: "load")) loaded)
                [] -> failure
            )
            requests
    evalIO (Bytes.writeFile path (Fixture.wire (prefix ++ suffix)))
    digest <- evalIO (Artifact.identity "Resident fixture transcript" path)
    let fields = ["format" .= format, "owner" .= identity selected, "loads" .= (loads :: [Value]), "result_sha256" .= digest]
        requestRelease = object ("action" .= ("release" :: Text) : fields)
        acknowledgement = object (["stage" .= ("released" :: Text), "measurement" .= measurement "released"] ++ fields)
    pure (Exchange (map fst requests) prefix suffix (Batch.permission (map fst requests)) requestRelease acknowledgement)

scenario :: [Exchange] -> Scenario
scenario = scenarioWith owner

scenarioWith :: Natural -> [Exchange] -> Scenario
scenarioWith selected exchanges = Scenario exchanges acknowledgement "IFS= read -r extra && exit 29\nexit 0"
  where
    acknowledgement = object ["stage" .= ("closed" :: Text), "format" .= format, "owner" .= identity selected, "groups" .= length exchanges, "measurement" .= measurement "closed"]

script :: FilePath -> Scenario -> String
script root selected = scriptWith (owner, replicate (length (groups selected)) adapter) root selected

scriptWith :: (Natural, [FilePath]) -> FilePath -> Scenario -> String
scriptWith (physical, paths) root selected =
    unlines
        ( [ "test \"$2\" = " ++ Serial.quote ("--session=" ++ show physical) ++ " || exit 20"
          , "printf '%s\\n' \"$$\" >> " ++ Serial.quote (root </> "pids")
          ]
            ++ concat (zipWith groupScript [0 :: Int ..] (zip paths (groups selected)))
            ++ [ receive "closing" (Bytes.init (Fixture.wire [object ["format" .= format, "owner" .= identity physical, "action" .= ("close" :: Text)]]))
               , "printf '%s\\n' closed > " ++ Serial.quote (root </> "closed")
               , emit [closed selected]
               , ending selected
               ]
        )
  where
    groupScript index (path, group) =
        [ receive "request" (Batch.input path (calls group))
        , emit (before group)
        , receive "permission" (permission group)
        , "printf '%s\\n' approved > " ++ Serial.quote (root </> ("approved" ++ show index))
        , emit (after group)
        , receive "release" (Bytes.init (Fixture.wire [release group]))
        , emit [released group]
        ]
    receive variable expected = "IFS= read -r " ++ variable ++ " || exit 21\ntest \"$" ++ variable ++ "\" = " ++ Serial.quote (Bytes.unpack expected) ++ " || exit 22"
    emit values = "printf '%s\\n' " ++ unwords (map (Serial.quote . Bytes.unpack) (Bytes.lines (Fixture.wire values)))

run :: FilePath -> Scenario -> PropertyT IO (Either Worker.Failure [Trajectory], ByteString)
run root selected = do
    let path = root </> "resident.sh"
        worker = Worker.Worker "/bin/sh" path root adapter [] Nothing
    evalIO (writeFile path (script root selected))
    buffer <- evalIO (newIORef [])
    let options = Resident.Options worker owner (Transcript.echoing (\line -> modifyIORef' buffer (line :)))
    returned <- evalIO (Resident.withResident options (\resident -> executeGroups resident (groups selected)))
    emitted <- evalIO (Bytes.unlines . reverse <$> readIORef buffer)
    pure (returned, emitted)

concluded :: FilePath -> String -> Scenario -> PropertyT IO (Either Worker.Failure [Trajectory], Maybe Transcript.Outcome)
concluded root body selected = do
    let path = root </> "resident.sh"
        worker = Worker.Worker "/bin/sh" path root adapter [] Nothing
    evalIO (writeFile path body)
    closing <- evalIO (newIORef Nothing)
    let transcript = Transcript.Transcript (const (pure ())) (const (pure ())) (writeIORef closing . Just)
    returned <- evalIO (Resident.withResident (Resident.Options worker owner transcript) (\resident -> executeGroups resident (groups selected)))
    ending <- evalIO (readIORef closing)
    pure (returned, ending)

interrupted :: FilePath -> Scenario -> ByteString -> PropertyT IO (Maybe String, ByteString, Maybe Transcript.Outcome)
interrupted root selected refused = do
    let path = root </> "resident.sh"
        worker = Worker.Worker "/bin/sh" path root adapter [] Nothing
    evalIO (writeFile path (script root selected))
    buffer <- evalIO (newIORef [])
    failed <- evalIO (newIORef False)
    closing <- evalIO (newIORef Nothing)
    let append bytes = modifyIORef' buffer (bytes :)
        write line = do
            already <- readIORef failed
            if line == refused && not already
                then writeIORef failed True >> append (Bytes.take 1 line) >> ioError (userError "The transcript refused a write")
                else append (Bytes.snoc line '\n')
        transcript = Transcript.Transcript write append (writeIORef closing . Just)
    thrown <- evalIO (tryIOError (Resident.withResident (Resident.Options worker owner transcript) (\resident -> executeGroups resident (groups selected))))
    emitted <- evalIO (Bytes.concat . reverse <$> readIORef buffer)
    ending <- evalIO (readIORef closing)
    pure (either (Just . ioeGetErrorString) (const Nothing) thrown, emitted, ending)

executeGroups :: Resident.Resident scope -> [Exchange] -> IO (Either Worker.Failure [Trajectory])
executeGroups _ [] = pure (Right [])
executeGroups resident (group : remaining) = do
    returned <- Resident.run resident adapter Nothing (calls group)
    case returned of
        Left problem -> pure (Left problem)
        Right values -> fmap (values ++) <$> executeGroups resident remaining
