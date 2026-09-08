{-# LANGUAGE OverloadedStrings #-}

module Store (store, workspace) where

import Control.Concurrent (forkFinally, newEmptyMVar, putMVar, readMVar, takeMVar)
import Control.Exception (try)
import Control.Monad (forM, forM_)
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as Text
import Data.Either (partitionEithers)
import Hedgehog
import Invar.Store qualified as S
import System.Directory (doesPathExist, getTemporaryDirectory)
import System.FilePath ((</>))
import System.IO.Error (isAlreadyExistsError, isDoesNotExistError)
import System.Posix.Files (createNamedPipe, createSymbolicLink, ownerModes)
import System.Posix.Temp (mkdtemp)

store :: Group
store =
    Group
        "Actual artifact publication"
        [ ("publication exposes the complete staged bytes", once publication)
        , ("conflict does not overwrite either artifact", once conflict)
        , ("missing staging cannot produce a receipt", once missing)
        , ("path aliases are rejected before publication", once names)
        , ("nonregular staging is not published", once nonregular)
        , ("concurrent publishers cannot replace the winner", once concurrent)
        ]
  where
    once = withTests 1 . property

workspace :: PropertyT IO FilePath
workspace = evalIO $ do
    temporary <- getTemporaryDirectory
    mkdtemp (temporary </> "invar-store.")

publication :: PropertyT IO ()
publication = do
    root <- workspace
    let target = S.Location root "staging" "policy"
        payload = Bytes.pack [minBound .. maxBound]
    evalIO (Bytes.writeFile (root </> "staging") payload)
    receipt <- evalIO (S.publish target)
    S.location receipt === target
    evalIO (Bytes.readFile (root </> "policy")) >>= (=== payload)
    evalIO (doesPathExist (root </> "staging")) >>= (=== False)

conflict :: PropertyT IO ()
conflict = do
    root <- workspace
    evalIO $ do
        Bytes.writeFile (root </> "staging") "new policy"
        Bytes.writeFile (root </> "policy") "committed policy"
    result <- evalIO (try (S.publish (S.Location root "staging" "policy")))
    case result of
        Left (S.Failure S.Rename problem) -> assert (isAlreadyExistsError problem)
        unexpected -> annotateShow unexpected >> failure
    evalIO (Bytes.readFile (root </> "policy")) >>= (=== "committed policy")
    evalIO (Bytes.readFile (root </> "staging")) >>= (=== "new policy")

missing :: PropertyT IO ()
missing = do
    root <- workspace
    result <- evalIO (try (S.publish (S.Location root "missing" "policy")))
    case result of
        Left (S.Failure S.Prepare problem) -> assert (isDoesNotExistError problem)
        unexpected -> annotateShow unexpected >> failure
    evalIO (doesPathExist (root </> "policy")) >>= (=== False)

names :: PropertyT IO ()
names = do
    root <- workspace
    forM_ ["", ".", "..", "../policy", "/policy", "policy\0suffix"] $ \name ->
        forM_ [S.Location root name "policy", S.Location root "staging" name] (reject S.Validate)
    reject S.Validate (S.Location root "policy" "policy")

reject :: S.Phase -> S.Location -> PropertyT IO ()
reject expected target = do
    result <- evalIO (try (S.publish target))
    case result of
        Left (S.Failure actual _) -> actual === expected
        unexpected -> annotateShow unexpected >> failure

nonregular :: PropertyT IO ()
nonregular = do
    root <- workspace
    evalIO $ do
        Bytes.writeFile (root </> "original") "original policy"
        createSymbolicLink "original" (root </> "alias")
        createNamedPipe (root </> "pipe") ownerModes
    forM_ ["alias", "pipe"] $ \name -> do
        reject S.Prepare (S.Location root name "policy")
        evalIO (doesPathExist (root </> "policy")) >>= (=== False)
    evalIO (Bytes.readFile (root </> "original")) >>= (=== "original policy")

concurrent :: PropertyT IO ()
concurrent = do
    root <- workspace
    let candidates = [("first", "first policy"), ("second", "second policy")]
    results <- evalIO $ do
        gate <- newEmptyMVar
        pending <- forM candidates $ \(name, payload) -> do
            Bytes.writeFile (root </> Text.unpack name) payload
            result <- newEmptyMVar
            _ <- forkFinally (readMVar gate >> try (S.publish (S.Location root name "policy"))) (putMVar result)
            pure result
        putMVar gate ()
        traverse takeMVar pending
    outcomes <- traverse evalEither results
    let (errors, receipts) = partitionEithers outcomes
    length receipts === 1
    length errors === 1
    forM_ errors $ \(S.Failure phase problem) -> do
        phase === S.Rename
        assert (isAlreadyExistsError problem)
    forM_ receipts $ \receipt -> do
        expected <- evalMaybe (lookup (S.staging (S.location receipt)) candidates)
        evalIO (Bytes.readFile (root </> "policy")) >>= (=== expected)
