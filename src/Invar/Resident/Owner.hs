{-# LANGUAGE OverloadedStrings #-}

module Invar.Resident.Owner (State, Released (..), Boundary.Owner (..), Boundary.Role (..), start, owner, groups, initial, release, close) where

import Control.Monad (foldM, unless, void, when)
import Data.Aeson (Object, Value (..))
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Set qualified as Set
import Data.Text (Text)
import Invar.Infer.Framing qualified as Framing
import Invar.Measurement.Duration qualified as Duration
import Invar.Resident qualified as Boundary
import Invar.Spec.Invocation qualified as V
import Numeric.Natural (Natural)

data State = State
    { owner :: Boundary.Owner
    , groups :: Natural
    , original :: [Framing.Frame]
    , model :: Maybe Object
    , identities :: (Set.Set V.CallId, Set.Set V.AttemptId, Set.Set V.Instance)
    , clock :: Maybe Duration.Clock
    }

data Released = Released {records :: [ByteString], bindings :: [V.Binding], loaded :: [Object], acknowledgement :: ByteString}

start :: Boundary.Owner -> State
start selected = State selected 0 [] Nothing (Set.empty, Set.empty, Set.empty) Nothing

initial :: State -> Bool
initial current = groups current == 0

release :: State -> Released -> Either String State
release current group = do
    frames <- if null (records group) then pure [] else Framing.decode (Bytes.unlines (records group))
    acknowledged <- Boundary.measured "released" (acknowledgement group)
    let (leading, execution) = span ((`elem` map Just preparation) . stage) frames
        retained = if initial current then leading else original current
        selected = case (model current, loaded group) of
            (Nothing, first : _) -> Just first
            (previous, _) -> previous
        expected = [Framing.fields record | record <- retained, stage record == Just "profile"] ++ maybe [] pure selected
    mapM_ (\actual -> mapM_ (\reference -> unless (all (\key -> Fields.lookup key reference == Fields.lookup key actual) ["model", "revision"]) (Left "Resident model or revision differs from its original physical load")) expected) (loaded group)
    when (null (bindings group)) (Left "Resident group requires a consumed invocation")
    admitted <- foldM fresh (identities current) (bindings group)
    mapM_ (\name -> when (length (filter ((== Just name) . stage) leading) > 1) (Left "Repeated resident operation measurement")) ["load", "activation"]
    measured <- traverse (\record -> Duration.admit (Framing.raw record) (Framing.fields record)) (filter ((`elem` map Just measuredStages) . stage) (leading ++ execution))
    chosen <- clocks (clock current) (measured ++ [acknowledged])
    pure current {groups = groups current + 1, original = retained, model = selected, identities = admitted, clock = chosen}

close :: State -> ByteString -> Either String ()
close current encoded = do
    elapsed <- Boundary.closed (owner current) (groups current) encoded
    void (clocks (clock current) [elapsed])

fresh :: (Set.Set V.CallId, Set.Set V.AttemptId, Set.Set V.Instance) -> V.Binding -> Either String (Set.Set V.CallId, Set.Set V.AttemptId, Set.Set V.Instance)
fresh (calls, attempts, instances) bound = do
    let call = V.boundCall bound
        attempt = V.boundAttempt bound
        instanceId = V.boundInstance bound
    when (Set.member call calls || Set.member attempt attempts || Set.member instanceId instances) (Left "Resident physical process reuses a historical invocation identity")
    pure (Set.insert call calls, Set.insert attempt attempts, Set.insert instanceId instances)

clocks :: Maybe Duration.Clock -> [Duration.Duration] -> Either String (Maybe Duration.Clock)
clocks expected measured = case maybe [] pure expected ++ map Duration.clock measured of
    [] -> pure Nothing
    selected : rest -> do
        unless (all (== selected) rest) (Left "Resident physical process mixes measurement clocks")
        pure (Just selected)

stage :: Framing.Frame -> Maybe Text
stage record = case Fields.lookup "stage" (Framing.fields record) of
    Just (String name) -> Just name
    _ -> Nothing

preparation :: [Text]
preparation = ["loading", "profile", "load", "activation"]

measuredStages :: [Text]
measuredStages = ["load", "activation", "inference", "reward_update", "artifacts", "checkpoint"]
