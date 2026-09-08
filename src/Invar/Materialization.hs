{-# LANGUAGE OverloadedStrings #-}

module Invar.Materialization (image, learning) where

import Crypto.Hash.SHA256 qualified as SHA256
import Data.ByteString.Char8 qualified as Bytes
import Invar.Artifact qualified as Artifact
import Invar.Spec.Load qualified as L

image :: (String, String, String, String) -> L.Image
image (adapter, tokenizer, base, assembly) = L.Image (Bytes.pack identity) (Bytes.pack assembly)
  where
    encoded = Bytes.intercalate "\0" (map Bytes.pack ["invar-inference-materialization-v1", adapter, tokenizer, base, assembly])
    identity = Artifact.hex (SHA256.hash encoded)

learning :: (String, String, String, String, String, String) -> L.Image
learning (policy, learner, tokenizer, base, assembly, reference) = L.Image (Bytes.pack identity) (Bytes.pack assembly)
  where
    encoded = Bytes.intercalate "\0" (map Bytes.pack ["invar-learning-materialization-v1", policy, learner, tokenizer, base, assembly, reference])
    identity = Artifact.hex (SHA256.hash encoded)
