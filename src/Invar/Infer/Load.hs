{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

module Invar.Infer.Load (Plan, Error (..), prepare, program, register, registerBatch) where

import Control.Monad (foldM, unless, when, (>=>))
import Data.Aeson (Object, eitherDecodeStrict, withObject, (.:))
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.Text.Encoding (encodeUtf8)
import Invar.Infer qualified as I
import Invar.Infer.Wire qualified as Wire
import Invar.Load qualified as Load
import Invar.Spec.Invocation qualified as V
import Invar.Spec.Load qualified as L

data Plan = Plan V.Binding Load.Plan

data Error = Loading Load.Error | Lifecycle V.Error | Registry L.Error | Protocol String
    deriving (Eq, Show)

data Registration = Serial | Batched

prepare :: V.Binding -> I.Plan -> Either Error Plan
prepare bound planned = Plan bound <$> either (Left . Loading) Right (Load.prepare bound (I.image (I.requested planned)))

program :: Plan -> ByteString
program (Plan _ loading) = Load.program loading

register :: Plan -> L.Registry -> ByteString -> Either Error L.Registry
register planned = registerMode (Serial, planned)

registerBatch :: L.Registry -> [(Plan, ByteString)] -> Either Error L.Registry
registerBatch registry selected = do
    unless (not (null selected) && null (L.active registry)) (Left (Protocol "A finite activation batch requires an idle load registry"))
    foldM (\current (planned, encoded) -> registerMode (Batched, planned) current encoded) registry selected

registerMode :: (Registration, Plan) -> L.Registry -> ByteString -> Either Error L.Registry
registerMode selected registry encoded = do
    events <- traverse (either (Left . Protocol) Right . eitherDecodeStrict) (Bytes.lines encoded)
    (seen, updated) <- foldM (advance selected) (False, registry) events
    unless seen (Left (Protocol "Missing completed policy load"))
    pure updated

advance :: (Registration, Plan) -> (Bool, L.Registry) -> Object -> Either Error (Bool, L.Registry)
advance (mode, planned) (seen, registry) value = do
    stage <- parse (.: "stage") value
    case stage :: String of
        "unloaded_adapter" -> do
            case mode of
                Batched -> Left (Protocol "A prepared activation batch cannot unload one of its members")
                Serial -> pure ()
            when seen (Left (Protocol "Unload follows the new policy load"))
            bound <- parse Wire.binding value
            previous <- registryError (L.historical registry (V.boundInstance bound))
            unless (V.completedBinding (L.report previous) == bound) (Left (Protocol "Unload binding differs from the live load"))
            programBytes <- encodeUtf8 <$> parse (.: "program") value
            unless (V.completedProgram (L.report previous) == programBytes) (Left (Protocol "Unload program differs from the live load"))
            updated <- registryError (L.unload (V.boundInstance bound) registry)
            pure (False, updated)
        "loaded_adapter" -> do
            when seen (Left (Protocol "Duplicate policy load within one invocation"))
            case mode of
                Serial -> unless (null (L.active registry)) (Left (Protocol "Previous policy load remains live"))
                Batched -> pure ()
            let Plan _ loading = planned
            updated <- either (Left . Loading) Right (Load.register loading value registry)
            pure (True, updated)
        "consumed" -> do
            actual <- parse ((.: "load") >=> withObject "consumed policy load" Load.invocation) value
            let Plan bound loading = planned
                encoded = Load.program loading
            unless (actual == (bound, encoded)) (Left (Protocol "Inference consumption names a different policy load"))
            pure (seen, registry)
        _ -> pure (seen, registry)

parse :: (Object -> Parser value) -> Object -> Either Error value
parse parser = either (Left . Protocol) Right . parseEither parser

registryError :: Either L.Error value -> Either Error value
registryError = either (Left . Registry) Right
