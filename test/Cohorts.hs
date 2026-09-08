{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}

module Cohorts (cohorts) where

import Control.Monad (forM_, join)
import Data.Aeson (Value (..), encode, object, (.=))
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Word (Word32)
import Hedgehog
import Invar.Cohort qualified as C
import Invar.Infer qualified as I
import Invar.Infer.Result qualified as R
import Invar.Reward qualified as Reward

cohorts :: Group
cohorts =
    Group
        "Scoped report admission"
        [ ("delivery order cannot select logical batch order", once ordered)
        , ("missing and repeated results cannot form a batch", once incomplete)
        , ("the declared rule computes the reward", once scoring)
        , ("each observation must match its member request", once mismatch)
        , ("the declaration fixes complete groups and one policy", once declaration)
        ]
  where
    once = withTests 1 . property

definition :: PropertyT IO C.Definition
definition = do
    first <- plan 17
    second <- plan 18
    expected <- evalEither (Reward.decimal "#### 12")
    pure C.Definition {C.policy = identity, C.tasks = [C.Task "first" "question" first expected, C.Task "second" "question" second expected]}

identity :: String
identity = replicate 64 'a'

plan :: Integer -> PropertyT IO I.Plan
plan seed = evalEither (I.prepare I.Request {I.artifact = identity, I.tokenizer = replicate 64 'c', I.base = replicate 64 'e', I.assembly = replicate 64 'f', I.prompt = "Compute the answer.", I.tokens = 2, I.temperature = 0.8, I.seed = seed})

report :: I.Plan -> String -> Bool -> PropertyT IO R.Result
report planned text limited = evalEither (R.observe planned (Bytes.unlines (map (Lazy.toStrict . encode) [loaded, output])))
  where
    input = I.requested planned
    requested = object ["prompt" .= I.prompt input, "tokens" .= I.tokens input, "temperature" .= I.temperature input, "seed" .= I.seed input]
    loaded = object ["stage" .= String "loaded_adapter", "requested" .= I.artifact input, "consumed" .= I.artifact input, "tokenizer" .= I.tokenizer input, "base" .= I.base input, "assembly" .= I.assembly input]
    output = object ["stage" .= String "result", "adapter" .= I.artifact input, "tokenizer" .= I.tokenizer input, "base" .= I.base input, "assembly" .= I.assembly input, "request" .= requested, "tokens" .= [1, 2, 3 :: Int], "prompt_length" .= Number 1, "behavior" .= [-0.5, -0.25 :: Double], "behavior_bits" .= [0xbf000000, 0xbe800000 :: Word32], "truncated" .= limited, "text" .= text]

exercise :: C.Definition -> (forall scope. C.Cohort scope -> PropertyT IO ()) -> PropertyT IO ()
exercise declared action = join (evalEither (C.withCohort declared action))

filled :: C.Member scope -> PropertyT IO (C.Observation scope)
filled member = report (C.planned member) "#### 12" False >>= evalEither . C.record member

ordered :: PropertyT IO ()
ordered = do
    declared <- definition
    exercise declared $ \cohort -> do
        supplied <- traverse filled (reverse (C.members cohort))
        batch <- evalEither (C.admit cohort supplied)
        map (I.seed . R.consumed . C.observed) (C.observations batch) === [17, 18]
        map C.reward (C.observations batch) === [1, 1]

rejected :: C.Error -> Either C.Error result -> PropertyT IO ()
rejected expected result = case result of
    Left actual -> actual === expected
    Right _ -> failure

incomplete :: PropertyT IO ()
incomplete = do
    declared <- definition
    exercise declared $ \cohort -> do
        supplied <- traverse filled (C.members cohort)
        rejected C.IncompleteCohort (C.admit cohort [])
        rejected C.IncompleteCohort (C.admit cohort (take 1 supplied))
        rejected C.RepeatedResult (C.admit cohort (supplied ++ supplied))

scoring :: PropertyT IO ()
scoring = do
    declared <- definition
    alternate <- evalEither (Reward.decimal "#### 13")
    exercise declared {C.tasks = map (\task -> task {C.rule = alternate}) (C.tasks declared)} $ \cohort -> do
        supplied <- traverse filled (C.members cohort)
        batch <- evalEither (C.admit cohort supplied)
        map C.reward (C.observations batch) === [0, 0]
        map (Reward.rule . C.scored) (C.observations batch) === [alternate, alternate]
        map (Reward.value . C.scored) (C.observations batch) === [0, 0]
        forM_ (C.members cohort) $ \member -> do
            output <- report (C.planned member) "#### 13" True
            scored <- evalEither (C.record member output)
            C.reward scored === 0

mismatch :: PropertyT IO ()
mismatch = do
    declared <- definition
    different <- plan 99
    output <- report different "#### 12" False
    exercise declared $ \cohort ->
        forM_ (zip (C.tasks declared) (C.members cohort)) $ \(task, member) ->
            rejected (C.RequestMismatch (C.name task)) (C.record member output)

declaration :: PropertyT IO ()
declaration = do
    declared <- definition
    let entries = C.tasks declared
        cases =
            [ (C.EmptyCohort, declared {C.tasks = []})
            , (C.DuplicateMember, declared {C.tasks = entries ++ entries})
            , (C.SingletonGroup, declared {C.tasks = take 1 entries})
            , (C.InvalidIdentity, declared {C.tasks = map (\task -> task {C.name = ""}) entries})
            , (C.PolicyMismatch "first", declared {C.policy = replicate 64 'b'})
            ]
    forM_ cases $ \(expected, changed) -> rejected expected (C.withCohort changed (const ()))
