{-# LANGUAGE OverloadedStrings #-}

module Invar.Qualification.Report (value, emit) where

import Data.Aeson (Value, encode, object, (.=))
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Invar.Spec.Evidence qualified as E
import Invar.Spec.Qualification qualified as Q
import System.IO (hFlush, stderr)

value :: Q.QualifiedResult -> Value
value qualified =
    object
        [ "format" .= ("invar-qualification-certificate/v1" :: String)
        , "judgement" .= (if null (E.assumptions accepted) then "unconditional" else "conditional" :: String)
        , "subject" .= object ["byte_encoding" .= ("latin1" :: String), "domain" .= Bytes.unpack domain, "binding" .= Bytes.unpack binding]
        , "conclusion" .= claim (E.conclusion accepted)
        , "assumptions" .= map claim (E.assumptions accepted)
        , "methods" .= map show (E.methods accepted)
        ]
  where
    accepted = Q.certificate qualified
    (domain, binding) = Q.bindings (Q.qualifiedSubject qualified)

claim :: E.Claim -> Value
claim (E.External obligation) =
    object
        [ "kind" .= ("external" :: String)
        , "predicate" .= E.predicate obligation
        , "reference" .= Bytes.unpack (E.specification obligation)
        , "observation" .= E.observation obligation
        , "domain_ref" .= ("subject.domain" :: String)
        , "binding_ref" .= ("subject.binding" :: String)
        ]
claim (E.OutputEqual completed expected) = object ["kind" .= ("output-equal" :: String), "completion" .= show completed, "expected" .= Bytes.unpack expected]
claim (E.All required) = object ["kind" .= ("all" :: String), "claims" .= map claim required]
claim (E.Implies antecedent consequent) = object ["kind" .= ("implies" :: String), "antecedent" .= claim antecedent, "consequent" .= claim consequent]

emit :: Maybe Q.QualifiedResult -> IO ()
emit Nothing = pure ()
emit (Just qualified) = do
    Lazy.hPut stderr (encode (value qualified))
    Bytes.hPutStrLn stderr Bytes.empty
    hFlush stderr
