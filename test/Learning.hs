{-# LANGUAGE OverloadedStrings #-}

module Learning (learning, world, learnerWorld) where

import Control.Monad (forM_)
import Data.Aeson qualified as J
import Data.Char (ord)
import Data.Either (isLeft)
import Data.Map.Strict qualified as Map
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Learn.Program qualified as P
import Invar.Learn.Wire qualified as W
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Program qualified as S
import Invar.Spec.Value (Scalar (..), Value (..))
import Numeric.Natural (Natural)
import Properties (campaign)

learning :: Group
learning = Group "Keyed learning emissions" [("lowering preserves the declared update request", once request), ("joint key renaming preserves the numerical payload", campaign renaming), ("one-source relabeling changes sample reward association", once association), ("groups order and probability roles cannot be substituted", once membership), ("interleaved samples retain their declared groups", once groups), ("source types reject non-bit behavior and operational inputs", once types), ("behavior bit patterns are not converted to numeric floats", once bits), ("load image must agree with each materialized input identity", once materialized), ("behavior representation is distinct from the learner load", once representations), ("reference scores are present exactly when the reference differs from the policy", once scored), ("optimizer steps split the logical order into consecutive mini-batches", once steps), ("samples come from version max(0, u - d) and need reference scores only when their behavior policy is not the reference", once versions)]
  where
    once = withTests 1 . property

record :: [(String, Value Natural)] -> Value Natural
record = Record . Map.fromList

text :: String -> Value Natural
text = Sequence . map (Atom . Token . fromIntegral . ord)

number :: Rational -> Value Natural
number = Atom . Number

generation :: Natural -> String -> Value Natural
generation version producer = record [("version", Atom (Token version)), ("policy", text producer)]

selector :: [Natural] -> Value Natural
selector keys = Mapping (Map.fromList [(key, Atom (Boolean True)) | key <- keys])

world :: E.World
world = Map.fromList [(S.Semantic "policy", image), (S.Semantic "learner", learner), (S.Semantic "reference", text (replicate 64 'c')), (S.Semantic "reference_source", text "engine"), (S.Semantic "algorithm", record [("epsilon", number (1 / 5)), ("penalty", number (1 / 25)), ("delta", number (1 / 10000)), ("steps", number 1)]), (S.Semantic "trajectories", Mapping (Map.fromList [(2, trajectory 17 12), (9, trajectory 18 13)])), (S.Semantic "behavior_model", record [("base", text (replicate 64 '0')), ("assembly", text (replicate 64 '1'))]), (S.Semantic "schedule", record [("update", Atom (Token 0)), ("staleness", Atom (Token 0))]), (S.Semantic "generations", Mapping (Map.fromList [(key, generation 0 (replicate 64 'a')) | key <- [2, 9]])), (S.Semantic "behavior", Mapping (Map.fromList [(2, Sequence [Atom (Bits32 0xbf800000)]), (9, Sequence [Atom (Bits32 0xbf000000)])])), (S.Semantic "reference_scores", Mapping (Map.fromList [(2, Sequence [Atom (Bits32 0xbfa00000)]), (9, Sequence [Atom (Bits32 0xbf400000)])])), (S.Semantic "rewards", Mapping (Map.fromList [(2, number 1), (9, number 0)])), (S.Semantic "groups", Sequence [selector [2, 9]]), (S.Semantic "order", Sequence [selector [9], selector [2]])]
  where
    image = record [("artifact", text "22f6619dc862f0f4c9bec5d1a1a6f11958b2299d02f18d1f04aa5e0e0b94d3d6"), ("profile", text (replicate 64 'f'))]
    learner = record [("policy", text (replicate 64 'a')), ("learner", text (replicate 64 'b')), ("tokenizer", text (replicate 64 'd')), ("base", text (replicate 64 'e')), ("assembly", text (replicate 64 'f')), ("optimizer", record [("learning_rate", number (1 / 500)), ("betas", Sequence [number (4 / 5), number (19 / 20)]), ("epsilon", number (1 / 10000000)), ("weight_decay", number (1 / 100))])]
    trajectory seed token = record [("prompt", text "Compute the answer."), ("seed", number seed), ("limit", Atom (Token 1)), ("temperature", number (4 / 5)), ("tokens", Sequence [Atom (Token 11), Atom (Token token)]), ("prompt_length", Atom (Token 1)), ("text", text "#### 12"), ("truncated", Atom (Boolean False))]

learnerWorld :: E.World
learnerWorld = Map.insert (S.Semantic "reference_source") (text "learner") world

expected :: J.Value
expected = J.object ["specification" J..= ("grpo-token-mean/v1" :: String), "policy" J..= replicate 64 'a', "learner" J..= replicate 64 'b', "tokenizer" J..= replicate 64 'd', "base" J..= replicate 64 'e', "assembly" J..= replicate 64 'f', "behavior_model" J..= J.object ["base" J..= replicate 64 '0', "assembly" J..= replicate 64 '1'], "schedule" J..= J.object ["update" J..= (0 :: Int), "staleness" J..= (0 :: Int)], "reference" J..= replicate 64 'c', "reference_source" J..= ("engine" :: String), "epsilon" J..= (0.2 :: Double), "penalty" J..= (0.04 :: Double), "delta" J..= (0.0001 :: Double), "optimizer" J..= optimizer, "order" J..= (["s0", "s1"] :: [String]), "steps" J..= [["s0", "s1" :: String]], "samples" J..= [sample "s0" (18, 13, 0xbf000000, 0), sample "s1" (17, 12, 0xbf800000, 1)]]
  where
    optimizer = J.object ["learning_rate" J..= (0.002 :: Double), "betas" J..= ([0.8, 0.95] :: [Double]), "epsilon" J..= (0.0000001 :: Double), "weight_decay" J..= (0.01 :: Double)]
    sample name (seed, token, word, reward) = J.object ["reference_bits" J..= [if word == 0xbf000000 then 0xbf400000 else 0xbfa00000 :: Integer], "sample" J..= (name :: String), "group" J..= ("g0" :: String), "prompt" J..= ("Compute the answer." :: String), "seed" J..= (seed :: Integer), "limit" J..= (1 :: Int), "temperature" J..= (0.8 :: Double), "tokens" J..= [11, token :: Int], "prompt_length" J..= (1 :: Int), "version" J..= (0 :: Int), "behavior_policy" J..= replicate 64 'a', "behavior_bits" J..= [word :: Integer], "text" J..= ("#### 12" :: String), "truncated" J..= False, "reward" J..= (reward :: Double), "advantage_bits" J..= (if reward == 0 then 0xbf7ff2e5 else 0x3f7ff2e5 :: Integer)]

emission :: E.World -> PropertyT IO E.Emission
emission inputs = do
    checked <- evalEither P.checked
    commands <- evalEither (A.run checked inputs)
    case commands of
        [command] -> pure command
        _ -> failure

lower :: E.World -> PropertyT IO J.Value
lower inputs = emission inputs >>= evalEither . W.lower

request :: PropertyT IO ()
request = lower world >>= (=== expected)

representations :: PropertyT IO ()
representations = do
    original <- emission world
    let different = Map.insert (S.Semantic "behavior_model") (record [("base", text (replicate 64 '2')), ("assembly", text (replicate 64 '3'))]) world
    changed <- emission different
    W.image changed === W.image original
    before <- evalEither (W.lower original) >>= json @(Map.Map String J.Value)
    after <- evalEither (W.lower changed) >>= json @(Map.Map String J.Value)
    Map.delete "behavior_model" before === Map.delete "behavior_model" after
    assert (Map.lookup "behavior_model" before /= Map.lookup "behavior_model" after)
    forM_ ["base", "assembly"] $ \name -> do
        let invalid = record [(field, text (if field == name then "invalid" else replicate 64 '0')) | field <- ["base", "assembly"]]
        command <- emission (Map.insert (S.Semantic "behavior_model") invalid world)
        case W.lower command of
            Left (W.Shape _) -> success
            unexpected -> annotateShow unexpected >> failure

rename :: (Natural -> Natural) -> Value Natural -> Value Natural
rename names value = case value of
    Mapping entries -> Mapping (Map.fromList [(names key, rename names payload) | (key, payload) <- Map.toList entries])
    Record fields -> Record (fmap (rename names) fields)
    Sequence values -> Sequence (map (rename names) values)
    Atom _ -> value

renaming :: PropertyT IO ()
renaming = do
    base <- forAll (Gen.integral (Range.linear 0 keyRange))
    reverseKeys <- forAll Gen.bool
    let names key = base + if (key == 2) == reverseKeys then 3 else 1
    cover 20 "key order reversed" reverseKeys
    cover 20 "key order preserved" (not reverseKeys)
    lower (fmap (rename names) world) >>= (=== expected)
  where
    keyRange = 10000

association :: PropertyT IO ()
association = do
    let changed = Map.adjust (rename (\key -> if key == 2 then 9 else 2)) (S.Semantic "rewards") world
    result <- lower changed
    assert (result /= expected)

groups :: PropertyT IO ()
groups = do
    let expanded = Map.mapWithKey expand world
        grouped = Map.insert (S.Semantic "groups") (Sequence [selector [2, 9], selector [12, 19]]) expanded
        ordered = Map.insert (S.Semantic "order") (Sequence (map (selector . pure) [19, 2, 12, 9])) grouped
    result <- lower ordered >>= json @(Map.Map String J.Value)
    entries <- evalMaybe (Map.lookup "samples" result) >>= json @[Map.Map String J.Value]
    map (Map.lookup "group") entries === map (Just . J.String) ["g1", "g0", "g1", "g0"]
  where
    expand source (Mapping values)
        | source `elem` map S.Semantic ["trajectories", "generations", "behavior", "reference_scores", "rewards"] = Mapping (Map.union values (Map.mapKeys (+ offset) values))
    expand _ value = value
    offset = 10

json :: (J.FromJSON value) => J.Value -> PropertyT IO value
json value = case J.fromJSON value of
    J.Success result -> pure result
    J.Error problem -> annotate problem >> failure

membership :: PropertyT IO ()
membership = forM_ cases $ \(source, value) -> do
    command <- emission (Map.insert (S.Semantic source) value world)
    case W.lower command of
        Left (W.Membership _) -> success
        unexpected -> annotateShow unexpected >> failure
  where
    cases = [("order", Sequence [selector [2], selector [2]]), ("order", Sequence [selector [2, 9]]), ("order", Sequence []), ("groups", Sequence [selector [2], selector [9]]), ("groups", Sequence [selector [2, 9], selector [2, 9]]), ("groups", Sequence []), ("groups", Sequence [Mapping (Map.fromList [(2, Atom (Boolean False)), (9, Atom (Boolean True))])]), ("rewards", Mapping (Map.singleton 2 (number 1))), ("behavior", Mapping (Map.singleton 9 (Sequence [Atom (Bits32 0xbf000000)])))]

types :: PropertyT IO ()
types = do
    checked <- evalEither P.checked
    let source = S.Semantic "behavior"
        kind = S.MapType (S.SequenceType S.BitsType)
        replaced = Mapping (Map.fromList [(2, Sequence [number (-1)]), (9, Sequence [number (-(1 / 2))])])
    A.run checked (Map.insert source replaced world) === Left (E.InvalidInput source kind)
    case A.run checked (Map.insert (S.Operational "history") (number 1) world) of
        Left (E.ExtraInputs _) -> success
        unexpected -> annotateShow unexpected >> failure

bits :: PropertyT IO ()
bits = do
    let changed bit = Map.insert (S.Semantic "behavior") (Mapping (Map.fromList [(2, Sequence [Atom (Bits32 bit)]), (9, Sequence [Atom (Bits32 0xbf000000)])])) world
    negative <- lower (changed 0x80000000)
    positive <- lower (changed 0)
    assert (negative /= positive)

materialized :: PropertyT IO ()
materialized = forM_ changed $ \inputs -> do
    command <- emission inputs
    case W.lower command of
        Left (W.Shape _) -> success
        unexpected -> annotateShow unexpected >> failure
  where
    changed =
        Map.insert (S.Semantic "reference") other world
            : [Map.adjust (replace name) (S.Semantic source) world | (source, name) <- fields]
    fields = [("policy", "artifact"), ("policy", "profile")] ++ [("learner", name) | name <- ["policy", "learner", "tokenizer", "base", "assembly"]]
    replace name (Record values) = Record (Map.insert name other values)
    replace _ value = value
    other = text (replicate 64 '0')

scored :: PropertyT IO ()
scored = do
    let scores values = Map.insert (S.Semantic "reference_scores") (Mapping (Map.fromList [(key, Sequence (map (Atom . Bits32) value)) | (key, value) <- values]))
        image = record [("artifact", text "ff31c2f196638cd68857b997da9b92144cca48ef3b386a1c5e31f203adacd011"), ("profile", text (replicate 64 'f'))]
        policy = Map.insert (S.Semantic "policy") image . Map.insert (S.Semantic "reference") (text (replicate 64 'a'))
        unscored = scores [(2, []), (9, [])]
    _ <- lower (policy (unscored world))
    forM_ [unscored world, policy world, scores [(2, [0xbfa00000, 0xbfa00000]), (9, [0xbf400000])] world, scores [(2, [0x3f800000]), (9, [0xbf400000])] world] $ \inputs -> do
        command <- emission inputs
        assert (isLeft (W.lower command))

steps :: PropertyT IO ()
steps = do
    let declared count = Map.insert (S.Semantic "algorithm") (record [("epsilon", number (1 / 5)), ("penalty", number (1 / 25)), ("delta", number (1 / 10000)), ("steps", number count)]) world
    two <- lower (declared 2) >>= json @(Map.Map String J.Value)
    Map.lookup "steps" two === Just (J.toJSON [["s0" :: String], ["s1"]])
    forM_ [0, 3, 1 / 2] $ \count -> do
        compiled <- emission (declared count)
        assert (isLeft (W.lower compiled))

versions :: PropertyT IO ()
versions = do
    let scheduled update staleness = Map.insert (S.Semantic "schedule") (record [("update", Atom (Token update)), ("staleness", Atom (Token staleness))])
        generated assigned = Map.insert (S.Semantic "generations") (Mapping (Map.fromList assigned))
        uniform version producer = generated [(key, generation version producer) | key <- [2, 9]]
        unscored = Map.insert (S.Semantic "reference_scores") (Mapping (Map.fromList [(key, Sequence []) | key <- [2, 9]]))
        reference = replicate 64 'c'
        other = replicate 64 'e'
    earlier <- lower (scheduled 1 1 (uniform 0 reference (unscored world))) >>= json @(Map.Map String J.Value)
    Map.lookup "schedule" earlier === Just (J.object ["update" J..= (1 :: Int), "staleness" J..= (1 :: Int)])
    entries <- evalMaybe (Map.lookup "samples" earlier) >>= json @[Map.Map String J.Value]
    map (Map.lookup "version") entries === replicate 2 (Just (J.toJSON (0 :: Int)))
    map (Map.lookup "behavior_policy") entries === replicate 2 (Just (J.toJSON reference))
    map (Map.lookup "reference_bits") entries === replicate 2 (Just (J.toJSON ([] :: [Int])))
    _ <- lower (scheduled 1 1 (uniform 0 other world))
    _ <- lower (scheduled 3 1 (uniform 2 other world))
    startup <- lower (scheduled 0 1 (uniform 0 (replicate 64 'a') world)) >>= json @(Map.Map String J.Value)
    started <- evalMaybe (Map.lookup "samples" startup) >>= json @[Map.Map String J.Value]
    map (Map.lookup "version") started === replicate 2 (Just (J.toJSON (0 :: Int)))
    let refused =
            [ scheduled 1 1 (uniform 0 reference world)
            , scheduled 1 1 (uniform 0 other (unscored world))
            , scheduled 2 1 (uniform 0 other world)
            , scheduled 1 0 (uniform 0 other world)
            , scheduled 0 0 (uniform 0 other world)
            , scheduled 1 1 (generated [(2, generation 0 other), (9, generation 1 other)] world)
            , scheduled 1 1 (generated [(2, generation 0 other)] world)
            , scheduled 1 1 (generated [(2, generation 0 other), (9, generation 0 (replicate 64 'd'))] world)
            , scheduled 0 1 (uniform 0 other world)
            ]
    forM_ refused $ \inputs -> do
        command <- emission inputs
        assert (isLeft (W.lower command))
