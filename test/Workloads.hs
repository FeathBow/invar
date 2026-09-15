{-# LANGUAGE OverloadedStrings #-}

module Workloads (workloads, declared, encoded, array, replace) where

import Control.Monad (forM_)
import Data.Aeson (Value (..), encode, object, toJSON, (.=))
import Data.Aeson.Key (Key)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.Foldable (toList)
import Data.Maybe (listToMaybe)
import Data.Text qualified as Text
import Hedgehog
import Invar.Workload qualified as Workload
import Numeric.Natural (Natural)
import Updates (alter, change, field)

workloads :: Group
workloads = Group "Shared frozen workload input" [("decoded tasks preserve logical inputs and exact byte identity", once identity), ("task and permutation inventories are checked before execution", once inventories), ("sampling and finite reward rules belong to the core", once sampling), ("integer JSON values do not depend on a Python runtime type", once numbers), ("duplicate keys and trailing JSON are rejected", once ambiguous)]
  where
    once = withTests 1 . property

declared :: Value
declared = toJSON [cohort [17, 29], cohort [17, 29, 43, 71]]
  where
    cohort seeds = object ["tasks" .= map task seeds, "order" .= [0 .. length seeds - 1], "delivery" .= reverse [0 .. length seeds - 1]]
    task seed = object ["name" .= ("sample/" ++ show seed), "group" .= String "question", "prompt" .= String "Compute one plus one.", "seed" .= (seed :: Integer), "tokens" .= (4 :: Natural), "temperature" .= (0.8 :: Double), "answer" .= String "#### 2"]

encoded :: Value -> ByteString
encoded = Lazy.toStrict . encode

array :: Value -> [Value]
array (Array values) = toList values
array _ = error "Expected a fixture array"

replace :: ByteString -> ByteString -> ByteString -> ByteString
replace old new original = let (prefix, suffix) = Bytes.breakSubstring old original in prefix <> new <> Bytes.drop (Bytes.length old) suffix

identity :: PropertyT IO ()
identity = do
    result <- evalEither (Workload.decode (encoded declared))
    alternate <- evalEither (Workload.decode (encoded declared <> "\n"))
    Workload.value result === declared
    Workload.value alternate === Workload.value result
    assert (Workload.digest result /= Workload.digest alternate)
    map (map Workload.name . Workload.tasks) (Workload.cycles result) === [["sample/17", "sample/29"], ["sample/17", "sample/29", "sample/43", "sample/71"]]
    map Workload.order (Workload.cycles result) === [[0, 1], [0, 1, 2, 3]]
    map Workload.delivery (Workload.cycles result) === [[1, 0], [3, 2, 1, 0]]

inventories :: PropertyT IO ()
inventories = do
    first <- evalMaybe (listToMaybe (array declared))
    task <- evalMaybe (listToMaybe (array (field "tasks" first)))
    let malformed = [toJSON ([] :: [Value]), object [], toJSON [change "tasks" (toJSON ([] :: [Value])) first], toJSON [change "tasks" (toJSON [task]) first], toJSON [change "tasks" (toJSON [task, task]) first]]
    mapM_ reject malformed
    forM_ ["order", "delivery"] $ \axis ->
        forM_ [toJSON [Number 0], toJSON [Number 0, Number 0], toJSON [Number 0, Number 2], toJSON [Bool False, Number 1], Null] $ \value ->
            reject (toJSON [change axis value first])
    forM_ ["name", "group"] $ \key -> reject (taskChange key (String ""))
    reject (taskChange "group" (String "singleton"))

sampling :: PropertyT IO ()
sampling = do
    forM_ [("tokens", Number 0), ("tokens", Number (-1)), ("tokens", Number 0.5), ("tokens", Bool True), ("temperature", Number 0), ("temperature", Bool True), ("temperature", String "NaN"), ("seed", Number 0.5), ("seed", Bool True), ("prompt", String (Text.singleton '\0')), ("answer", String "wrong"), ("extra", Null)] $ \(key, value) -> reject (taskChange key value)
    accepted <- evalEither (Workload.decode (encoded (taskChange "prompt" (String ""))))
    first <- evalMaybe (listToMaybe (Workload.cycles accepted))
    task <- evalMaybe (listToMaybe (Workload.tasks first))
    Workload.prompt task === ""
    case Workload.decode (replace "\"temperature\":0.8" "\"temperature\":1e400" (encoded declared)) of
        Left _ -> success
        Right unexpected -> annotateShow unexpected >> failure

numbers :: PropertyT IO ()
numbers = do
    let changed = replace "\"tokens\":4" "\"tokens\":4.0" (replace "\"seed\":17" "\"seed\":17e0" (encoded declared))
    accepted <- evalEither (Workload.decode changed)
    map (map Workload.seed . Workload.tasks) (Workload.cycles accepted) === [[17, 29], [17, 29, 43, 71]]
    map (map Workload.tokens . Workload.tasks) (Workload.cycles accepted) === [[4, 4], [4, 4, 4, 4]]

ambiguous :: PropertyT IO ()
ambiguous = forM_ [encoded declared <> " null", replace "\"name\":" "\"name\":\"ignored\",\"name\":" (encoded declared), replace "\"tasks\":" "\"tasks\":[],\"tasks\":" (encoded declared)] $ \input ->
    case Workload.decode input of
        Left _ -> success
        Right unexpected -> annotateShow unexpected >> failure

taskChange :: Key -> Value -> Value
taskChange key value = toJSON (alter 0 first (array declared))
  where
    first cohort = change "tasks" (toJSON (alter 0 (change key value) (array (field "tasks" cohort)))) cohort

reject :: Value -> PropertyT IO ()
reject input = case Workload.decode (encoded input) of
    Left _ -> success
    Right unexpected -> annotateShow unexpected >> failure
