{-# LANGUAGE Safe #-}

module Invar.Spec.Measurement (
    Field (..),
    Binding (..),
    Orientation (..),
    MethodSpec (..),
    Method,
    Measurement,
    Error (..),
    prepare,
    specification,
    measure,
    value,
    raw,
    inputs,
    emission,
    method,
) where

import Control.Monad (unless, when)
import Data.ByteString (ByteString)
import Data.Char (isSpace)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Invar.Spec.Artifact qualified as Artifact
import Invar.Spec.Evaluate qualified as Evaluate
import Invar.Spec.Program qualified as Program
import Invar.Spec.Value (Scalar (..), Value (..))

data Field = Response | Truncated | Tokens | BehaviorWords | PromptLength | Prompt | Seed | Horizon
    deriving (Eq, Ord, Show, Enum, Bounded)

data Binding = ObservedField Field | Parameter String
    deriving (Eq, Show)

data Orientation = IncreasingLoss | DecreasingLoss
    deriving (Eq, Show)

data MethodSpec = MethodSpec
    { program :: ByteString
    , bindings :: Map Program.Source Binding
    , sink :: String
    , outputSpecification :: String
    , lower :: Rational
    , upper :: Rational
    , orientation :: Orientation
    , meaning :: String
    }
    deriving (Eq, Show)

data Method = Method MethodSpec Artifact.Checked

instance Eq Method where
    first == second = specification first == specification second

instance Show Method where
    show = show . specification

data Measurement = Measurement MethodSpec Evaluate.World Evaluate.Emission Rational Rational
    deriving (Eq, Show)

data Error
    = InvalidMethod String
    | InvalidProgram Artifact.LoadError
    | InvalidInputs Evaluate.Error
    | BindingInventoryMismatch
    | UnexpectedOutput [Evaluate.Emission]
    | OutsideRange Rational Rational Rational
    deriving (Eq, Show)

prepare :: MethodSpec -> Either Error Method
prepare spec = do
    unless (lower spec < upper spec) (Left (InvalidMethod "measurement range must have positive width"))
    when (any (all isSpace) [sink spec, outputSpecification spec, meaning spec]) (Left (InvalidMethod "measurement declaration must name its output and meaning"))
    checked <- either (Left . InvalidProgram) Right (Artifact.load (program spec))
    pure (Method spec checked)

specification :: Method -> MethodSpec
specification (Method spec _) = spec

measure :: Method -> Evaluate.World -> Either Error Measurement
measure (Method spec checked) world = do
    unless (Map.keysSet world == Map.keysSet (bindings spec)) (Left BindingInventoryMismatch)
    outputs <- either (Left . InvalidInputs) Right (Artifact.run checked world)
    case outputs of
        [observed@(Evaluate.Emission name role (Atom (Number number)))]
            | name == sink spec && role == outputSpecification spec -> do
                unless (number >= lower spec && number <= upper spec) (Left (OutsideRange (lower spec) (upper spec) number))
                let normalized = case orientation spec of
                        IncreasingLoss -> (number - lower spec) / (upper spec - lower spec)
                        DecreasingLoss -> (upper spec - number) / (upper spec - lower spec)
                pure (Measurement spec world observed number normalized)
        _ -> Left (UnexpectedOutput outputs)

value, raw :: Measurement -> Rational
value (Measurement _ _ _ _ normalized) = normalized
raw (Measurement _ _ _ number _) = number

inputs :: Measurement -> Evaluate.World
inputs (Measurement _ world _ _ _) = world

emission :: Measurement -> Evaluate.Emission
emission (Measurement _ _ output _ _) = output

method :: Measurement -> MethodSpec
method (Measurement spec _ _ _ _) = spec
