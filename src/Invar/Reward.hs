{-# LANGUAGE Safe #-}

module Invar.Reward (Rule, Scored, Error (..), decimal, expected, score, value, rule, program, inputs, emission) where

import Data.ByteString (ByteString)
import Data.Char (ord)
import Data.Map.Strict qualified as Map
import Invar.Reward.Decimal qualified as Decimal
import Invar.Reward.Program qualified as Program
import Invar.Spec.Artifact qualified as A
import Invar.Spec.Evaluate qualified as E
import Invar.Spec.Program qualified as P
import Invar.Spec.Value (Scalar (..), Value (..))

newtype Rule = Rule Rational
    deriving (Eq, Show)

data Scored = Scored
    { declaredRule :: Rule
    , programBytes :: ByteString
    , consumedInputs :: E.World
    , emittedResult :: E.Emission
    , rewardValue :: Rational
    }
    deriving (Eq, Show)

data Error = InvalidAnswer | ProgramError A.LoadError | EvaluationError E.Error | UnexpectedEmission [E.Emission]
    deriving (Eq, Show)

decimal :: String -> Either Error Rule
decimal = maybe (Left InvalidAnswer) (Right . Rule) . Decimal.parse

expected :: Rule -> Rational
expected (Rule answer) = answer

score :: Rule -> String -> Bool -> Either Error Scored
score selected@(Rule answer) text truncated = do
    checked <- either (Left . ProgramError) Right Program.checked
    let supplied = Map.fromList [(P.Semantic "rule", Atom (Number answer)), (P.Semantic "response", characters text), (P.Semantic "truncated", Atom (Boolean truncated))]
    outputs <- either (Left . EvaluationError) Right (A.run checked supplied)
    case outputs of
        [result@(E.Emission "score" "exact-decimal/v1" (Atom (Number reward)))] ->
            Right Scored {declaredRule = selected, programBytes = A.bytes checked, consumedInputs = supplied, emittedResult = result, rewardValue = reward}
        _ -> Left (UnexpectedEmission outputs)
  where
    characters = Sequence . map (Atom . Token . fromIntegral . ord)

value :: Scored -> Rational
value = rewardValue

rule :: Scored -> Rule
rule = declaredRule

program :: Scored -> ByteString
program = programBytes

inputs :: Scored -> E.World
inputs = consumedInputs

emission :: Scored -> E.Emission
emission = emittedResult
