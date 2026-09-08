{-# LANGUAGE OverloadedStrings #-}

module Rewards (rewards) where

import Control.Monad (forM_)
import Data.ByteString.Char8 qualified as Bytes
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Hedgehog
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Reward qualified as R
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Dependency qualified as D
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Program qualified as P
import Invar.Spec.Value (Scalar (..), Value (..))

rewards :: Group
rewards =
    Group
        "Declared decimal rewards"
        [ ("reward uses exact decimal equality", once exact)
        , ("invalid answer specifications are rejected", once invalid)
        , ("only the final response line can score", once finalLine)
        , ("truncation and malformed responses score zero", once malformed)
        , ("a score retains its exact program inputs and emission", once retained)
        , ("the declared answer is a consumed semantic source", once declared)
        , ("operational response and control sources are rejected", once operational)
        , ("finite decimal programs agree with exact generated answers", withTests 100 (property generated))
        ]
  where
    once = withTests 1 . property

exact :: PropertyT IO ()
exact = do
    expected <- evalEither (R.decimal "#### 12")
    forM_ ["#### 12", "#### +12.00", "#### 00012.000", "\xa0#### 12\x3000"] $ \text ->
        evaluate expected text False >>= (=== 1)
    large <- evalEither (R.decimal "#### 9007199254740992")
    evaluate large "#### 9007199254740993" False >>= (=== 0)
    negative <- evalEither (R.decimal "#### -0.00001")
    evaluate negative "#### -0.0000100" False >>= (=== 1)
    zero <- evalEither (R.decimal "#### -0")
    evaluate zero "#### +0.000" False >>= (=== 1)

invalid :: PropertyT IO ()
invalid = forM_ ["", "12", "#### .5", "#### 12.", "#### 1e1", "#### NaN", "#### 1,000", "#### １２", "####  12", "####\t12"] $ \text ->
    R.decimal text === Left R.InvalidAnswer

finalLine :: PropertyT IO ()
finalLine = do
    expected <- evalEither (R.decimal "\x1f#### 12\x202f")
    forM_ ['\n', '\r', '\v', '\f', '\x1c', '\x1d', '\x1e', '\x85', '\x2028', '\x2029'] $ \separator ->
        evaluate expected ("Reasoning." ++ [separator] ++ "#### 12\r\n") False >>= (=== 1)
    evaluate expected "#### 12\nLater text" False >>= (=== 0)
    evaluate expected "Reasoning.\n  #### 12" False >>= (=== 0)

malformed :: PropertyT IO ()
malformed = do
    expected <- evalEither (R.decimal "#### 12")
    evaluate expected "#### 12" True >>= (=== 0)
    forM_ ["", "12", "#### 11", "#### 1e1", "#### NaN", "#### 1,000"] $ \text ->
        evaluate expected text False >>= (=== 0)

evaluate :: R.Rule -> String -> Bool -> PropertyT IO Rational
evaluate expected text truncated = R.value <$> evalEither (R.score expected text truncated)

retained :: PropertyT IO ()
retained = do
    expected <- evalEither (R.decimal "#### 12")
    result <- evalEither (R.score expected "Reasoning.\n#### 12" False)
    R.rule result === expected
    R.emission result === E.Emission "score" "exact-decimal/v1" (Atom (Number 1))
    Map.keysSet (R.inputs result) === Set.fromList [P.Semantic "rule", P.Semantic "response", P.Semantic "truncated"]
    Map.lookup (P.Semantic "rule") (R.inputs result) === Just (Atom (Number 12))
    Map.lookup (P.Semantic "truncated") (R.inputs result) === Just (Atom (Boolean False))
    checked <- evalEither (A.load (R.program result))
    evalEither (A.run checked (R.inputs result)) >>= (=== [R.emission result])

declared :: PropertyT IO ()
declared = do
    firstRule <- evalEither (R.decimal "#### 12")
    secondRule <- evalEither (R.decimal "#### 13")
    first <- evalEither (R.score firstRule "#### 12" False)
    second <- evalEither (R.score secondRule "#### 12" False)
    R.program first === R.program second
    R.value first === 1
    R.value second === 0
    assert (R.rule first /= R.rule second)
    checked <- evalEither (A.load (R.program first))
    let changed = Map.insert (P.Semantic "rule") (Atom (Number 13)) (R.inputs first)
    evalEither (A.run checked changed) >>= (=== [R.emission second])
    R.value first === 1

operational :: PropertyT IO ()
operational = do
    expected <- evalEither (R.decimal "#### 12")
    result <- evalEither (R.score expected "#### 12" False)
    forM_ [("response", "(sequence token)"), ("truncated", "bool")] $ \(name, kind) -> do
        extended <- replace "(sources " ("(sources ((operational \"outside\") " <> kind <> ") ") (R.program result)
        changed <- replace ("(input (semantic \"" <> name <> "\"))") "(input (operational \"outside\"))" extended
        case A.load changed of
            Left (A.ValidationError problem) -> problem === D.ForbiddenSources (Set.singleton (P.Operational "outside"))
            _ -> failure

replace :: Bytes.ByteString -> Bytes.ByteString -> Bytes.ByteString -> PropertyT IO Bytes.ByteString
replace needle replacement original = do
    let (prefix, suffix) = Bytes.breakSubstring needle original
    assert (not (Bytes.null suffix))
    pure (prefix <> replacement <> Bytes.drop (Bytes.length needle) suffix)

generated :: PropertyT IO ()
generated = do
    whole <- forAll (Gen.integral (Range.linear 0 maximumWhole))
    places <- forAll (Gen.int (Range.linear 1 maximumPlaces))
    fraction <- forAll (Gen.integral (Range.linear 0 (decimalBase ^ places - 1)))
    negative <- forAll Gen.bool
    let suffix = show (fraction :: Integer)
        sign = if negative then "-" else "+"
        answer = "#### " ++ sign ++ show (whole :: Integer) ++ "." ++ replicate (places - length suffix) '0' ++ suffix
    expected <- evalEither (R.decimal answer)
    evaluate expected ("\t" ++ answer ++ "\r\n ") False >>= (=== 1)
    evaluate expected (answer ++ "1") False >>= (=== 0)
    evaluate expected ("Reasoning.\n " ++ answer) False >>= (=== 0)
    evaluate expected answer True >>= (=== 0)
  where
    decimalBase = 10
    maximumWhole = decimalBase ^ (18 :: Int)
    maximumPlaces = 12
