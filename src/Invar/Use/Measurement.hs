module Invar.Use.Measurement (
    Field (..),
    Binding (..),
    Orientation (..),
    MethodSpec (..),
    Method,
    Measurement,
    Error (..),
    prepare,
    specification,
    value,
    raw,
    InputError (..),
    observe,
) where

import Data.Char (ord)
import Data.Map.Strict qualified as Map
import Invar.Infer qualified as Infer
import Invar.Infer.Result qualified as Result
import Invar.Spec.Domain qualified as Domain
import Invar.Spec.Measurement
import Invar.Spec.Measurement qualified as Measurement
import Invar.Spec.Value (Scalar (..), Value (..))
import Numeric.Natural (Natural)

data InputError = MissingParameter String | EvaluationFailed Error
    deriving (Eq, Show)

observe :: Method -> Domain.Input -> Result.Result -> Either InputError Measurement
observe selected input result = do
    world <- traverse resolve (bindings (specification selected))
    either (Left . EvaluationFailed) Right (Measurement.measure selected world)
  where
    resolve (Parameter name) = maybe (Left (MissingParameter name)) Right (Map.lookup name (Domain.parameters input))
    resolve (ObservedField field) = pure (observed field result)

observed :: Field -> Result.Result -> Value Natural
observed field result = case field of
    Response -> characters (Result.response result)
    Truncated -> Atom (Boolean (Result.truncated result))
    Tokens -> Sequence (map (Atom . Token) (Result.tokens result))
    BehaviorWords -> Sequence (map (Atom . Bits32) (Result.behaviorBits result))
    PromptLength -> Atom (Token (Result.promptLength result))
    Prompt -> characters (Infer.prompt requested)
    Seed -> Atom (Number (fromInteger (Infer.seed requested)))
    Horizon -> Atom (Token (Infer.tokens requested))
  where
    requested = Result.consumed result
    characters = Sequence . map (Atom . Token . fromIntegral . ord)
