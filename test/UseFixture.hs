{-# LANGUAGE OverloadedStrings #-}

module UseFixture (Trial (..), trials, fixture, repeated, workloadValue, run) where

import Calls (request)
import Data.Aeson (Value (..), encode, object, toJSON, (.=))
import Data.ByteString.Lazy qualified as Lazy
import Hedgehog
import Invar.Infer qualified as Infer
import Invar.Numerical qualified as N
import Invar.Spec.Invocation qualified as Invocation
import Invar.Use qualified as U
import Invar.Use.Decimal qualified as Decimal
import Invar.Workload qualified as Workload
import Logs qualified
import Numeric.Natural (Natural)

data Trial = Trial
    { name :: String
    , prompt :: String
    , seed :: Integer
    , answer :: String
    , before :: String
    , after :: String
    , truncated :: Bool
    }
    deriving (Eq, Show)

trials :: [Trial]
trials =
    [ Trial "a1" "question a" 1 "#### 12" "#### 12" "#### 0" False
    , Trial "a2" "question a" 2 "#### 12" "#### 12" "#### 12" False
    , Trial "a3" "question a" 3 "#### 12" "#### 12" "#### 12" False
    , Trial "b1" "question b" 1 "#### 12" "#### 0" "#### 0" False
    ]

workloadValue :: [Trial] -> Value
workloadValue values = toJSON [object ["tasks" .= map task values, "order" .= inventory, "delivery" .= inventory]]
  where
    inventory = [0 .. length values - 1]
    task value = object ["name" .= name value, "group" .= ("declared-group" :: String), "prompt" .= prompt value, "tokens" .= Infer.tokens request, "temperature" .= Infer.temperature request, "seed" .= seed value, "answer" .= answer value]

fixture :: [Trial] -> PropertyT IO U.BoundRun
fixture = repeated 0

repeated :: Natural -> [Trial] -> PropertyT IO U.BoundRun
repeated count values = do
    document <- evalEither (Workload.decode (Lazy.toStrict (encode (workloadValue values))))
    observations <- traverse paired (zip [0 ..] values)
    evalEither (Decimal.bind document observations)
  where
    paired (index, value) = do
        reference <- run (2 * index) N.Reference value
        candidate <- run (2 * index + 1) N.Candidate value
        repeats <- traverse (\offset -> run (100 + 10 * index + offset) N.Candidate value) [1 .. count]
        pure (U.Case (U.Key 0 (name value)) (N.BoundRun reference candidate) repeats)

run :: Natural -> N.Side -> Trial -> PropertyT IO N.Run
run identity side trial = do
    let requested = request {Infer.prompt = prompt trial, Infer.seed = seed trial}
        binding = Invocation.Binding (Invocation.CallId identity) (Invocation.AttemptId identity) (Invocation.Instance identity)
    planned <- evalEither (Infer.prepare requested)
    encoded <- Logs.single planned binding (case side of N.Reference -> before trial; N.Candidate -> after trial, truncated trial)
    pure (N.Run planned binding 0 encoded Nothing)
