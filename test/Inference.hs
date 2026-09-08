{-# LANGUAGE OverloadedStrings #-}

module Inference (inference) where

import Control.Monad (forM_)
import Hedgehog
import Invar.Infer qualified as I

inference :: Group
inference = Group "Checked inference arguments" [("semantic output supplies exact worker arguments", once arguments), ("invalid requests cannot produce executable plans", once invalid)]
  where
    once = withTests 1 . property

request :: I.Request
request = I.Request {I.artifact = replicate 64 'a', I.tokenizer = replicate 64 'c', I.base = replicate 64 'e', I.assembly = replicate 64 'f', I.prompt = "--seed=0\nA 'quoted' λ prompt; $(false)", I.tokens = 32, I.temperature = 0.8, I.seed = -1}

arguments :: PropertyT IO ()
arguments = do
    planned <- evalEither (I.prepare request)
    I.arguments planned === ["--digest=" ++ replicate 64 'a', "--tokenizer-digest=" ++ replicate 64 'c', "--base-digest=" ++ replicate 64 'e', "--assembly-digest=" ++ replicate 64 'f', "--prompt=--seed=0\nA 'quoted' λ prompt; $(false)", "--tokens=32", "--temperature=0.8", "--seed=-1"]
    let next = request {I.artifact = replicate 64 'b', I.prompt = "", I.tokens = 1, I.temperature = 1.25, I.seed = 17}
    changed <- evalEither (I.prepare next)
    I.arguments changed === ["--digest=" ++ replicate 64 'b', "--tokenizer-digest=" ++ replicate 64 'c', "--base-digest=" ++ replicate 64 'e', "--assembly-digest=" ++ replicate 64 'f', "--prompt=", "--tokens=1", "--temperature=1.25", "--seed=17"]

invalid :: PropertyT IO ()
invalid = forM_ cases $ \input -> case I.prepare input of
    Left (I.InvalidRequest _) -> success
    unexpected -> annotateShow unexpected >> failure
  where
    cases = [request {I.base = ""}, request {I.assembly = replicate 64 'F'}, request {I.tokenizer = ""}, request {I.tokenizer = replicate 64 'C'}, request {I.artifact = ""}, request {I.artifact = replicate 64 'A'}, request {I.tokens = 0}, request {I.temperature = 0}, request {I.temperature = -1}, request {I.temperature = 0 / 0}, request {I.temperature = 1 / 0}, request {I.prompt = "a\0b"}]
