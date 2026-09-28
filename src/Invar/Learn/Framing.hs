{-# LANGUAGE OverloadedStrings #-}

module Invar.Learn.Framing (readiness, completion) where

import Control.Monad (unless, void, when)
import Data.Aeson (Value (..))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Invar.Infer.Framing (stageName)
import Invar.Infer.Framing qualified as Frame
import Invar.Learn.Trace qualified as Trace
import Invar.Measurement.Duration qualified as Duration

readiness :: Bool -> Value -> ByteString -> Either String ()
readiness initial request encoded = do
    records <- Frame.decode encoded
    let (prefix, pending) = span (\record -> stageName record `elem` map (Just . String) ["loading", "profile", "load", "activation"]) records
        expected = if initial then [["load", "activation"], ["loading", "profile", "load", "activation"]] else [["activation"]]
    unless (map stageName prefix `elem` map (map (Just . String)) expected) (Left "Expected one initial learner load and one activation per update")
    mapM_ noPhase prefix
    mapM_ timing (filter (\record -> stageName record `elem` map (Just . String) ["load", "activation"]) prefix)
    void (Trace.readiness request (map Frame.fields pending))

completion :: Value -> ByteString -> Either String ()
completion request encoded = do
    records <- Frame.decode encoded
    case dropWhile ((/= Just (String "reward_update")) . stageName) (dropWhile ((/= Just (String "consumed")) . stageName) records) of
        updated : remaining@(_ : _) -> do
            unless (stageName updated == Just (String "reward_update")) (Left "Expected one actual reward update measurement")
            let (staging, ending) = span (\record -> stageName record `elem` map (Just . String) ["artifacts", "checkpoint"]) remaining
            mapM_ timing (updated : staging)
            case ending of
                [result] -> Trace.completion request (Frame.fields result)
                _ -> Left "Expected one measured and completed resident update"
        _ -> Left "Expected one measured and completed resident update"

noPhase :: Frame.Frame -> Either String ()
noPhase record = when (Fields.member "phase" (Frame.fields record)) (Left "Unexpected phase in learner loading observations")

timing :: Frame.Frame -> Either String ()
timing record = void (Duration.admit (Frame.raw record) (Frame.fields record))
