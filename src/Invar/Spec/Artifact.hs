{-# LANGUAGE Safe #-}

module Invar.Spec.Artifact (Checked, LoadError (..), encode, load, bytes, run) where

import Control.DeepSeq (force)
import Control.Monad (unless)
import Data.ByteString (ByteString)
import Data.Map.Strict qualified as Map
import Data.Set (Set)
import Data.Set qualified as Set
import Invar.Spec.Decode qualified as Decode
import Invar.Spec.Dependency qualified as D
import Invar.Spec.Encode qualified as Encode
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Operator qualified as O
import Invar.Spec.Program
import Invar.Spec.Syntax qualified as Syntax

data Checked = Checked ByteString E.Semantics [Command]

data LoadError
    = SyntaxError String
    | ValidationError D.Error
    | MissingMeaning String
    | ExtraMeanings (Set String)
    | MeaningMismatch String Signature Signature
    deriving (Eq, Show)

encode :: E.Semantics -> [Command] -> ByteString
encode meaning = Syntax.render . Encode.document meaning

load :: ByteString -> Either LoadError Checked
load encoded = do
    term <- either (Left . SyntaxError) Right (Syntax.parse encoded)
    decoded <- either (Left . SyntaxError) Right (Decode.document term)
    let finite = force decoded
        (schema, operations, program) = finite
        meaning = E.Semantics schema operations
    finite `seq` checkMeanings meaning
    either (Left . ValidationError) Right (D.checkSchema schema)
    _ <- either (Left . ValidationError) Right (D.checkCommands schema program)
    pure (Checked encoded meaning program)

checkMeanings :: E.Semantics -> Either LoadError ()
checkMeanings meaning = do
    let extra = Map.keysSet (E.meanings meaning) Set.\\ Map.keysSet (primitives (E.schema meaning))
    unless (Set.null extra) (Left (ExtraMeanings extra))
    mapM_ check (Map.toList (primitives (E.schema meaning)))
  where
    check (name, declared) = do
        operation <- maybe (Left (MissingMeaning name)) Right (Map.lookup name (E.meanings meaning))
        let actual = O.signature operation
        unless (actual == declared) (Left (MeaningMismatch name declared actual))

bytes :: Checked -> ByteString
bytes (Checked encoded _ _) = encoded

run :: Checked -> E.World -> Either E.Error [E.Emission]
run (Checked _ meaning program) inputs = E.runCommands meaning inputs program
