{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Framing (Released (..), readiness, completion, finite, resident) where

import Control.Monad (unless, void, when)
import Data.Aeson (Value (..), (.:))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseEither)
import Data.ByteString (ByteString)
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import Invar.Infer.Framing (stageName)
import Invar.Infer.Framing qualified as Frame
import Invar.Learn qualified as Learn
import Invar.Learn.Trace qualified as Trace
import Invar.Measurement.Duration qualified as Duration
import Invar.Resident qualified as Boundary
import Invar.Resident.Owner qualified as Owner
import Invar.Spec.Invocation qualified as V

data Released = Released Owner.State [Frame.Frame] Frame.Frame [Frame.Frame]

readiness :: Bool -> Value -> ByteString -> Either String ()
readiness initial request encoded = do
    records <- Frame.decode encoded
    let (prefix, pending) = span (\record -> stageName record `elem` map (Just . String) ["loading", "profile", "load", "activation"]) records
        expected = if initial then [["load", "activation"], ["loading", "profile", "load", "activation"]] else [["activation"]]
    unless (map stageName prefix `elem` map (map (Just . String)) expected) (Left "Expected one initial learner load and one activation per update")
    mapM_ noPhase prefix
    mapM_ timing (filter (\record -> stageName record `elem` map (Just . String) ["load", "activation"]) prefix)
    void (Trace.readiness request (map Frame.fields pending))

loadingPrefix :: [Frame.Frame] -> Either String ([Frame.Frame], [Frame.Frame])
loadingPrefix records = do
    let (prefix, rest) = span (\record -> stageName record `elem` map (Just . String) ["loading", "profile", "load"]) records
    unless (map stageName prefix `elem` map (map (Just . String)) [["load"], ["loading", "profile", "load"]]) (Left "Missing, duplicated or reordered model-loading observations")
    mapM_ noPhase prefix
    mapM_ timing (filter ((== Just (String "load")) . stageName) prefix)
    case rest of
        loaded : _ -> mapM_ (\profiled -> unless (all (\key -> Fields.lookup key (Frame.fields profiled) == Fields.lookup key (Frame.fields loaded)) ["model", "revision"]) (Left "Model profile differs from the loaded model or revision")) (filter ((== Just (String "profile")) . stageName) prefix)
        [] -> pure ()
    pure (prefix, rest)

completion :: ByteString -> Either String ()
completion encoded = do
    records <- Frame.decode encoded
    case dropWhile ((/= Just (String "reward_update")) . stageName) (dropWhile ((/= Just (String "consumed")) . stageName) records) of
        updated : remaining@(_ : _) -> do
            unless (stageName updated == Just (String "reward_update")) (Left "Expected one actual reward update measurement")
            let (staging, ending) = span (\record -> stageName record `elem` map (Just . String) ["artifacts", "checkpoint"]) remaining
            mapM_ timing (updated : staging)
            case ending of
                [result] -> Trace.completion (Frame.fields result)
                _ -> Left "Expected one measured and completed resident update"
        _ -> Left "Expected one measured and completed resident update"

finite :: Learn.Settings -> (V.Binding, Text, Value) -> [Frame.Frame] -> Either String ([Frame.Frame], Trace.Attempt)
finite settings declared records = do
    (prefix, events) <- loadingPrefix records
    when (null events) (Left "Missing learner execution after the model load")
    attempted <- Trace.attempt settings declared (map Frame.fields events)
    pure (prefix, either (\problem -> attempted {Trace.result = Nothing, Trace.stopped = Just problem}) (const attempted) (mapM_ timing (filter (staged ["reward_update", "artifacts", "checkpoint"]) events)))

resident :: Learn.Settings -> Owner.State -> (V.Binding, Text, Value) -> [Frame.Frame] -> Either String (Trace.Attempt, Either String Released)
resident settings physical declared@(bound, _, request) records = do
    let (leading, remaining) = span (staged ["loading", "profile", "load", "activation"]) records
        (execution, rest) = case break (staged ["result"]) remaining of
            (preceding, finished : after) -> (preceding ++ [finished], after)
            (preceding, []) -> (preceding, [])
        (ready, consumption) = break (staged ["consumed"]) execution
    readiness (Owner.initial physical) request (Frame.encode (leading ++ ready ++ take 1 consumption))
    attempted <- Trace.attempt settings declared (map Frame.fields execution)
    let released = case (Trace.result attempted, rest, execution, consumption) of
            (Just _, acknowledged : after, loaded : _, consumed : _) -> do
                completion (Frame.encode (leading ++ execution))
                loads <- parseEither (.: "load") (Frame.fields consumed)
                void (Boundary.observeRelease (Owner.owner physical, [loads], Frame.encode (leading ++ execution)) (Frame.raw acknowledged))
                next <- Owner.release physical (Owner.Released (map Frame.raw (leading ++ execution)) [bound] [Frame.fields loaded] (Frame.raw acknowledged))
                pure (Released next (leading ++ execution) acknowledged after)
            (Just _, _, _, _) -> Left "Resident group has no release acknowledgement"
            (Nothing, _, _, _) -> Left (fromMaybe "Incomplete resident update result" (Trace.stopped attempted))
    pure (attempted, released)

staged :: [Text] -> Frame.Frame -> Bool
staged names record = stageName record `elem` map (Just . String) names

noPhase :: Frame.Frame -> Either String ()
noPhase record = when (Fields.member "phase" (Frame.fields record)) (Left "Unexpected phase in learner loading observations")

timing :: Frame.Frame -> Either String ()
timing record = void (Duration.admit (Frame.raw record) (Frame.fields record))
