{-# LANGUAGE OverloadedStrings #-}

module Invar.Evidence.Encoding (verdict, premise) where

import Data.Aeson (Value (..), object, (.=))
import Data.ByteString (ByteString)
import Data.Text.Encoding qualified as Text
import Invar.Artifact qualified as Artifact
import Invar.Spec.Evidence qualified as Evidence

verdict :: (Evidence.Witness -> Value) -> Evidence.Verdict -> Value
verdict witness result = case result of
    Evidence.Accept certificate -> object ["status" .= ("accept" :: String), "assumptions" .= map premise (Evidence.assumptions certificate), "methods" .= Evidence.methodNames certificate]
    Evidence.Refute counterexample -> object ["status" .= ("refute" :: String), "witness" .= witness (Evidence.witness counterexample)]
    Evidence.Unknown problem -> object ["status" .= ("unknown" :: String), "reason" .= show problem]

premise :: Evidence.Claim -> Value
premise (Evidence.External obligation) =
    object
        [ "predicate" .= Evidence.predicate obligation
        , "specification" .= encoded (Evidence.specification obligation)
        , "observation" .= Evidence.observation obligation
        , "scope" .= Artifact.hex (Evidence.domain obligation)
        , "source_binding" .= encoded (Evidence.binding obligation)
        ]
premise other = object ["claim" .= show other]

encoded :: ByteString -> Value
encoded bytes = case Text.decodeUtf8' bytes of
    Right text -> String text
    Left _ -> object ["encoding" .= ("hex" :: String), "bytes" .= Artifact.hex bytes]
