{-# LANGUAGE OverloadedStrings #-}

module Invar.Replay.Update.Artifacts (compare) where

import Control.Monad (unless)
import Data.Aeson (Object, object, withObject, (.=))
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (Pair, parseEither)
import Invar.Artifact qualified as Artifact
import Invar.Learn.Report qualified as Report
import Invar.Policy.File qualified as File
import System.FilePath ((</>))
import Prelude hiding (compare)

compare :: (Report.Report, FilePath) -> (Report.Report, FilePath) -> IO [Pair]
compare (actual, staged) (expected, published) = do
    same <- File.withFile (staged </> "gradients.safetensors") $ \left ->
        File.withFile (published </> "gradients.safetensors") $ \right -> do
            verify "Replayed gradients" (Report.gradient actual) =<< File.rawIdentity left
            verify "Reference gradients" (Report.gradient expected) =<< File.rawIdentity right
            File.sameRepresentation left right
    probability "Replayed probabilities" actual (staged </> "probabilities.json")
    probability "Reference probabilities" expected (published </> "probabilities.json")
    leftFields <- result actual
    rightFields <- result expected
    policy <- either invalid pure (Report.artifact "adapter" actual)
    learner <- either invalid pure (Report.artifact "learner" actual)
    let compared = [(key, Fields.lookup key leftFields == Fields.lookup key rightFields) | key <- ["adapter", "learner", "probabilities", "update"]] ++ [("gradients", same)]
    pure
        [ "result_equal" .= all snd compared
        , "equal_fields" .= object [key .= equal | (key, equal) <- compared]
        , "gradients_file_digest_equal" .= (Report.gradient actual == Report.gradient expected)
        , "adapter" .= policy
        , "learner" .= learner
        ]
  where
    result :: Report.Report -> IO Object
    result = either invalid pure . parseEither (withObject "admitted update result" pure) . Report.result
    probability label observed path = do
        expectedDigest <- either invalid pure (Report.artifact "probabilities" observed)
        verify label expectedDigest =<< Artifact.identity label path

verify :: String -> String -> String -> IO ()
verify label expected actual = unless (expected == actual) (invalid (label ++ " file differs from its reported digest"))

invalid :: String -> IO value
invalid = ioError . userError
