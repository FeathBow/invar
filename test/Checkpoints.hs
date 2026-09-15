{-# LANGUAGE OverloadedStrings #-}

module Checkpoints (checkpoints) where

import Control.Concurrent (forkFinally, newEmptyMVar, putMVar, readMVar, takeMVar)
import Control.Exception (try)
import Control.Monad (forM, forM_, when)
import Data.ByteString (ByteString)
import Data.ByteString qualified as Bytes
import Data.ByteString.Char8 qualified as Text
import Data.Either (partitionEithers)
import Data.String (fromString)
import Hedgehog
import Invar.Store qualified as S
import Store (workspace)
import System.Directory (createDirectory, doesPathExist)
import System.FilePath ((</>))
import System.IO.Error (isAlreadyExistsError, isDoesNotExistError)
import System.Posix.Files (createNamedPipe, createSymbolicLink, ownerModes, readSymbolicLink)

checkpoints :: Group
checkpoints =
    Group
        "Actual checkpoint publication"
        (concatMap cases [S.RenameExclusive, S.LinkImmutable])
  where
    once = withTests 1 . property
    cases chosen =
        [ (fromString (show chosen ++ ": " ++ name), once (verify chosen))
        | (name, verify) <-
            [ ("all checkpoint members have one published name", publication)
            , ("any missing member prevents publication", incomplete)
            , ("even an empty destination cannot be replaced", conflict)
            , ("a dangling destination cannot be replaced", dangling)
            , ("checkpoint and member aliases are rejected", aliases)
            , ("concurrent publication preserves one complete checkpoint", concurrent)
            ]
        ]

stage :: FilePath -> ByteString -> IO ()
stage path value = do
    createDirectory path
    Bytes.writeFile (path </> "adapter.safetensors") ("adapter:" <> value)
    Bytes.writeFile (path </> "learner.pt") ("learner:" <> value)
    Bytes.writeFile (path </> "policy.json") ("policy:" <> value)

members :: [FilePath]
members = ["adapter.safetensors", "learner.pt", "policy.json"]

contents :: FilePath -> IO [ByteString]
contents path = traverse (Bytes.readFile . (path </>)) members

expected :: ByteString -> [ByteString]
expected value = map (<> value) ["adapter:", "learner:", "policy:"]

publication :: S.Method -> PropertyT IO ()
publication chosen = do
    root <- workspace
    let target = S.Location root "staging" "checkpoint"
    evalIO (stage (root </> "staging") "complete")
    receipt <- evalIO (S.publishCheckpoint chosen target)
    S.location receipt === target
    S.method receipt === chosen
    evalIO (contents (root </> "checkpoint")) >>= (=== expected "complete")
    evalIO (doesPathExist (root </> "staging")) >>= (=== (chosen == S.LinkImmutable))
    when (chosen == S.LinkImmutable) $ do
        evalIO (readSymbolicLink (root </> "checkpoint")) >>= (=== "staging")
        evalIO (contents (root </> "staging")) >>= (=== expected "complete")

incomplete :: S.Method -> PropertyT IO ()
incomplete chosen = forM_ members $ \missing -> do
    root <- workspace
    evalIO $ do
        createDirectory (root </> "staging")
        forM_ (filter (/= missing) members) $ \present ->
            Bytes.writeFile (root </> "staging" </> present) "present"
    result <- evalIO (try (S.publishCheckpoint chosen (S.Location root "staging" "checkpoint")))
    case result of
        Left (S.Failure S.Prepare problem) -> assert (isDoesNotExistError problem)
        unexpected -> annotateShow unexpected >> failure
    evalIO (doesPathExist (root </> "checkpoint")) >>= (=== False)
    forM_ (filter (/= missing) members) $ \present ->
        evalIO (Bytes.readFile (root </> "staging" </> present)) >>= (=== "present")

conflict :: S.Method -> PropertyT IO ()
conflict chosen = do
    root <- workspace
    evalIO $ do
        stage (root </> "staging") "new"
        createDirectory (root </> "checkpoint")
    result <- evalIO (try (S.publishCheckpoint chosen (S.Location root "staging" "checkpoint")))
    case result of
        Left (S.Failure phase problem) -> phase === mutation chosen >> assert (isAlreadyExistsError problem)
        unexpected -> annotateShow unexpected >> failure
    evalIO (contents (root </> "staging")) >>= (=== expected "new")
    evalIO (doesPathExist (root </> "checkpoint" </> "adapter.safetensors")) >>= (=== False)

mutation :: S.Method -> S.Phase
mutation S.RenameExclusive = S.Rename
mutation S.LinkImmutable = S.Link

dangling :: S.Method -> PropertyT IO ()
dangling chosen = do
    root <- workspace
    evalIO $ do
        stage (root </> "staging") "new"
        createSymbolicLink "absent" (root </> "checkpoint")
    result <- evalIO (try (S.publishCheckpoint chosen (S.Location root "staging" "checkpoint")))
    case result of
        Left (S.Failure phase problem) -> phase === mutation chosen >> assert (isAlreadyExistsError problem)
        unexpected -> annotateShow unexpected >> failure
    evalIO (readSymbolicLink (root </> "checkpoint")) >>= (=== "absent")
    evalIO (contents (root </> "staging")) >>= (=== expected "new")

reject :: S.Method -> S.Location -> PropertyT IO ()
reject chosen target = do
    result <- evalIO (try (S.publishCheckpoint chosen target))
    case result of
        Left (S.Failure S.Prepare _) -> success
        unexpected -> annotateShow unexpected >> failure
    evalIO (doesPathExist (S.directory target </> Text.unpack (S.destination target))) >>= (=== False)

aliases :: S.Method -> PropertyT IO ()
aliases chosen = do
    root <- workspace
    evalIO $ do
        stage (root </> "original") "original"
        createSymbolicLink "original" (root </> "alias")
    reject chosen (S.Location root "alias" "checkpoint")
    evalIO (contents (root </> "original")) >>= (=== expected "original")
    forM_ members $ \name ->
        forM_ [False, True] (rejectMember chosen name)

rejectMember :: S.Method -> FilePath -> Bool -> PropertyT IO ()
rejectMember chosen name pipe = do
    root <- workspace
    let neighbor = if name == "adapter.safetensors" then "learner.pt" else "adapter.safetensors"
    evalIO $ do
        createDirectory (root </> "staging")
        forM_ (filter (/= name) members) $ \present ->
            Bytes.writeFile (root </> "staging" </> present) "neighbor"
        if pipe
            then createNamedPipe (root </> "staging" </> name) ownerModes
            else createSymbolicLink neighbor (root </> "staging" </> name)
    reject chosen (S.Location root "staging" "checkpoint")
    evalIO (Bytes.readFile (root </> "staging" </> neighbor)) >>= (=== "neighbor")

concurrent :: S.Method -> PropertyT IO ()
concurrent chosen = do
    root <- workspace
    outcomes <- evalIO $ do
        gate <- newEmptyMVar
        pending <- forM ["first", "second"] $ \name -> do
            stage (root </> Text.unpack name) name
            result <- newEmptyMVar
            _ <- forkFinally (readMVar gate >> try (S.publishCheckpoint chosen (S.Location root name "checkpoint"))) (putMVar result)
            pure result
        putMVar gate ()
        traverse takeMVar pending
    completed <- traverse evalEither outcomes
    let (errors, receipts) = partitionEithers completed
    length receipts === 1
    length errors === 1
    forM_ errors $ \(S.Failure phase problem) -> do
        phase === mutation chosen
        assert (isAlreadyExistsError problem)
    forM_ receipts $ \receipt -> do
        S.method receipt === chosen
        let winner = S.staging (S.location receipt)
            loser = if winner == "first" then "second" else "first"
        evalIO (contents (root </> "checkpoint")) >>= (=== expected winner)
        evalIO (contents (root </> Text.unpack loser)) >>= (=== expected loser)
