{-# LANGUAGE OverloadedStrings #-}

module ResidentWorkloads (Fixture (..), setup, options, initial, writeScripts, memberCount, execution, arrival, ownerRoot) where

import BatchCalls qualified as Serial
import Calls qualified
import Control.Monad (forM_)
import Data.Aeson (Value (..), eitherDecodeStrict, object, toJSON, (.=))
import Data.ByteString.Char8 qualified as Bytes
import Data.List.NonEmpty (NonEmpty)
import Data.List.NonEmpty qualified as NonEmpty
import Hedgehog
import Invar.Cohort qualified as Cohort
import Invar.Infer qualified as Infer
import Invar.Infer.Invocation qualified as Call
import Invar.Reward qualified as Reward
import Invar.Rollout qualified as Rollout
import Invar.Spec.Load qualified as Load
import Invar.Worker qualified as Worker
import Numeric.Natural (Natural)
import ResidentFixture qualified as Resident
import System.Directory (createDirectory)
import System.FilePath ((</>))

data Fixture = Fixture {workloads :: NonEmpty Rollout.Options, owners :: [(FilePath, Resident.Scenario)]}

options :: Fixture -> [Rollout.Options]
options = NonEmpty.toList . workloads

initial :: Fixture -> Rollout.Options
initial = NonEmpty.head . workloads

memberCount :: Natural
memberCount = 3

execution :: [Natural]
execution = [2, 0, 1]

arrival :: [Natural]
arrival = [1, 2, 0]

ownerRoot :: FilePath -> Int -> FilePath
ownerRoot root slot = root </> ("owner" ++ show slot)

setup :: FilePath -> Int -> [String] -> PropertyT IO Fixture
setup root count policies = do
    plans <- traverse (evalEither . Infer.prepare . (\policy -> Calls.request {Infer.artifact = policy})) policies
    prepared <- traverse (settings root overlays) (zip [0 :: Int ..] plans)
    declared <- evalMaybe (NonEmpty.nonEmpty prepared)
    selected <- traverse (owner plans) [0 .. count - 1]
    let fixture = Fixture declared selected
    writeScripts fixture
    pure fixture
  where
    overlays = [[("CUDA_VISIBLE_DEVICES", show slot)] | slot <- [0 .. count - 1]]
    assigned slot = [index | (position, index) <- zip [0 :: Int ..] execution, position `mod` count == slot]
    owner plans slot = do
        let location = ownerRoot root slot
            members = assigned slot
        evalIO (createDirectory location)
        groups <- if null members then pure [] else traverse (group location (fromIntegral slot, members)) (zip [0 ..] plans)
        pure (location, Resident.scenarioWith (fromIntegral slot) groups)

settings :: FilePath -> [[(String, String)]] -> (Int, Infer.Plan) -> PropertyT IO Rollout.Options
settings root overlays (index, planned) = do
    expected <- evalEither (Reward.decimal "#### 12")
    let worker = Worker.Worker "/bin/sh" (root </> "driver.sh") root (root </> ("checkpoint" ++ show index) </> "adapter.safetensors") [] (Just "native configuration.json")
        tasks = [Cohort.Task ("member" ++ show member) "group" planned expected | member <- [0 .. memberCount - 1]]
    pure (Rollout.Options worker Rollout.Resident overlays (Cohort.Definition (Infer.artifact (Infer.requested planned)) tasks) execution arrival Nothing)

group :: FilePath -> (Natural, [Natural]) -> (Natural, Infer.Plan) -> PropertyT IO Resident.Exchange
group root (slot, members) (index, planned) = do
    (_, events) <- Calls.setup
    requests <- traverse (\member -> Serial.prepared planned events (index * memberCount + member) Nothing >>= reselect planned) members
    Resident.prepareWith root (slot, index) requests

reselect :: Infer.Plan -> (Call.Call, [Value]) -> PropertyT IO (Call.Call, [Value])
reselect planned (call, [loaded, consumed, result]) = do
    envelope <- evalEither (eitherDecodeStrict (Call.batchInput call))
    let requested = Infer.requested planned
        digest = toJSON (Infer.artifact requested)
        image = Infer.image requested
        loadedImage = object ["artifact" .= Bytes.unpack (Load.artifact image), "profile" .= Bytes.unpack (Load.profile image)]
        activation = Calls.change "image" loadedImage (Calls.change "requested" digest (Calls.change "consumed" digest loaded))
        consumption = Calls.change "program" (Calls.field "program" envelope) (Calls.change "adapter" digest consumed)
    pure (call, [activation, consumption, Calls.change "adapter" digest result])
reselect _ _ = failure

writeScripts :: Fixture -> PropertyT IO ()
writeScripts fixture = do
    let first = initial fixture
        paths = map (Worker.adapter . Rollout.worker) (options fixture)
        branch slot (root, scenario) = show slot ++ ")\n" ++ Resident.scriptWith (fromIntegral slot, paths) root scenario ++ "\n;;"
        header = "test \"$3\" = " ++ Serial.quote "--config=native configuration.json" ++ " || exit 20"
        body = unlines ([header, "case \"$CUDA_VISIBLE_DEVICES\" in"] ++ zipWith branch [0 :: Int ..] (owners fixture) ++ ["*) exit 31;;", "esac"])
    forM_ (owners fixture) $ \(_, scenario) -> assert (length (Resident.groups scenario) <= length paths)
    evalIO (writeFile (Worker.script (Rollout.worker first)) body)
