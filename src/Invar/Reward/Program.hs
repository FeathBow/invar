{-# LANGUAGE Safe #-}

module Invar.Reward.Program (checked) where

import Data.Char (ord)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Invar.Reward.Decimal qualified as Decimal
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Operator qualified as O
import Invar.Spec.Program qualified as P
import Invar.Spec.Value (Scalar (..), Value (..))

data Phase = Hash0 | Hash1 | Hash2 | Hash3 | Space | Begin | Signed | Whole | Point | Fraction | Invalid
    deriving (Enum)

checked :: Either A.LoadError A.Checked
checked = A.load (A.encode meaning [P.Emit "score" "exact-decimal/v1" result])
  where
    result = P.If (input "truncated") (number 0) (P.Project folded "reward")
    folded = P.Collect (P.FoldSequence (P.SequenceFold (input "response") "character" "previous" body initial))
    sources = Map.fromList [(P.Semantic "rule", P.NumberType), (P.Semantic "response", P.SequenceType P.TokenType), (P.Semantic "truncated", P.BooleanType)]
    operations = Map.fromList [("add", O.Add), ("multiply", O.Multiply), ("negate", O.Negate), ("and", O.And), ("not", O.Not), ("number-equal", O.Equal P.NumberType), ("token-equal", O.Equal P.TokenType)]
    sink = P.Sink "exact-decimal/v1" P.NumberType (Map.keysSet sources) Set.empty
    meaning = E.Semantics (P.Schema sources (O.signature <$> operations) (Map.singleton "score" sink)) operations

body :: P.Expr
body =
    P.Let "digit" digit
        $ P.Let "whitespace" (characters Decimal.whitespace)
        $ P.Let "line-break" (characters Decimal.lineBreaks)
        $ P.If (both (operation "not" [field "previous" "started"]) (P.Variable "whitespace")) (P.Variable "previous")
        $ P.Let "parsed" (P.If (P.Variable "line-break") parser advance)
        $ record [("parser", P.Variable "parsed"), ("started", boolean True), ("reward", P.If (P.Variable "whitespace") (field "previous" "reward") scored)]

initial :: P.Expr
initial = record [("parser", parser), ("started", boolean False), ("reward", number 0)]

parser :: P.Expr
parser = record [("phase", phase Hash0), ("value", number 0), ("scale", tenth), ("negative", boolean False)]

digit :: P.Expr
digit = record [("valid", characters alphabet), ("value", choices [(character letter, number value) | (letter, value) <- zip alphabet [0 ..]] (number 0))]
  where
    alphabet = "0123456789"

advance :: P.Expr
advance = record [("phase", nextPhase), ("value", nextValue), ("scale", nextScale), ("negative", nextSign)]
  where
    nextScale = P.If (both (field "digit" "valid") (stages [Point, Fraction])) (operation "multiply" [old "scale", tenth]) (old "scale")
    nextSign = P.If (both (stages [Begin]) (character '-')) (boolean True) (old "negative")

nextPhase :: P.Expr
nextPhase = choices transitions (phase Invalid)
  where
    transitions =
        [ (stages [Hash0, Hash1, Hash2, Hash3], P.If (character '#') (operation "add" [old "phase", number 1]) (phase Invalid))
        , (stages [Space], P.If (character ' ') (phase Begin) (phase Invalid))
        , (stages [Begin], P.If (characters "+-") (phase Signed) (requireDigit Whole))
        , (stages [Signed], requireDigit Whole)
        , (stages [Whole], P.If (field "digit" "valid") (phase Whole) (P.If (character '.') (phase Point) (phase Invalid)))
        , (stages [Point, Fraction], requireDigit Fraction)
        ]
    requireDigit selected = P.If (field "digit" "valid") (phase selected) (phase Invalid)

nextValue :: P.Expr
nextValue = P.If (field "digit" "valid") value (old "value")
  where
    value =
        choices
            [ (stages [Begin, Signed, Whole], operation "add" [operation "multiply" [old "value", number (fromInteger Decimal.decimalBase)], field "digit" "value"])
            , (stages [Point, Fraction], operation "add" [old "value", operation "multiply" [field "digit" "value", old "scale"]])
            ]
            (old "value")

scored :: P.Expr
scored = P.If valid (P.If matches (number 1) (number 0)) (number 0)
  where
    valid = anyOf [operation "number-equal" [field "parsed" "phase", phase selected] | selected <- [Whole, Fraction]]
    actual = P.If (field "parsed" "negative") (operation "negate" [field "parsed" "value"]) (field "parsed" "value")
    matches = operation "number-equal" [actual, input "rule"]

stages :: [Phase] -> P.Expr
stages selected = anyOf [operation "number-equal" [old "phase", phase value] | value <- selected]

characters :: String -> P.Expr
characters = anyOf . map character

character :: Char -> P.Expr
character value = operation "token-equal" [P.Variable "character", P.Constant P.TokenType (Atom (Token (fromIntegral (ord value))))]

choices :: [(P.Expr, P.Expr)] -> P.Expr -> P.Expr
choices = flip (foldr (uncurry P.If))

anyOf :: [P.Expr] -> P.Expr
anyOf values = choices [(value, boolean True) | value <- values] (boolean False)

both :: P.Expr -> P.Expr -> P.Expr
both first second = operation "and" [first, second]

field :: String -> String -> P.Expr
field name = P.Project (P.Variable name)

old :: String -> P.Expr
old = P.Project (field "previous" "parser")

record :: [(String, P.Expr)] -> P.Expr
record = P.Fields . Map.fromList

operation :: String -> [P.Expr] -> P.Expr
operation = P.Primitive

input :: String -> P.Expr
input = P.Read . P.Input . P.Semantic

number :: Rational -> P.Expr
number = P.Constant P.NumberType . Atom . Number

boolean :: Bool -> P.Expr
boolean = P.Constant P.BooleanType . Atom . Boolean

phase :: Phase -> P.Expr
phase = number . fromIntegral . fromEnum

tenth :: P.Expr
tenth = number (recip (fromInteger Decimal.decimalBase))
