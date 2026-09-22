{-# LANGUAGE OverloadedStrings #-}

module Invar.Use.Observation (Case (..), BoundRun (..), ObservationError (..), observe) where

import Control.Monad (foldM, unless)
import Crypto.Hash.SHA256 qualified as SHA256
import Data.Bifunctor (first)
import Data.Char (isSpace)
import Data.Foldable (toList)
import Data.List.NonEmpty (NonEmpty (..))
import Data.List.NonEmpty qualified as NonEmpty
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text qualified as Text
import Data.Text.Encoding qualified as Text
import Invar.Infer qualified as Infer
import Invar.Infer.Observation qualified as Inference
import Invar.Infer.Result qualified as Result
import Invar.Numerical qualified as Numerical
import Invar.Spec.Domain qualified as Domain
import Invar.Spec.Invocation qualified as Invocation
import Invar.Spec.Numerical qualified as N
import Invar.Spec.Use qualified as U
import Invar.Use.Measurement qualified as Measurement
import Numeric.Natural (Natural)

data Case = Case {caseKey :: U.Key, paired :: Numerical.BoundRun, repeats :: [Numerical.Run]}

data BoundRun = BoundRun {domain :: Domain.Domain, measurement :: Maybe Measurement.Method, cases :: [Case]}

data ObservationError
    = EmptyDomain
    | InvalidDomain String
    | DuplicateInput U.Key
    | DuplicateCase U.Key
    | InventoryMismatch [U.Key] [U.Key]
    | InvalidPair U.Key Numerical.ObservationError
    | InvalidRepeat U.Key Natural Numerical.ObservationError
    | InvalidResponse U.Key N.Side String
    | InputMismatch U.Key String
    | MixedImplementation N.Side U.Key
    | ReusedExecution N.Side U.Key
    | MeasurementFailed U.Key N.Side Measurement.InputError
    deriving (Eq, Show)

observe :: BoundRun -> Either ObservationError U.Observed
observe supplied = do
    let document = domain supplied
        declared = Map.fromList [(Domain.inputKey item, item) | item <- toList (Domain.declaredInputs document)]
        provided = cases supplied
    validateDomain document
    uniqueCases provided
    let actual = Map.fromList [(caseKey item, (paired item, repeats item)) | item <- provided]
    unless (Map.keysSet declared == Map.keysSet actual) (Left (InventoryMismatch (Map.keys declared) (Map.keys actual)))
    matched <- traverse (checked (measurement supplied)) (Map.toAscList (Map.intersectionWith (,) declared actual))
    samples <- maybe (Left EmptyDomain) Right (NonEmpty.nonEmpty matched)
    mapM_ (`consistent` samples) [N.Reference, N.Candidate]
    let identity = SHA256.hash (Text.encodeUtf8 (Text.pack (show (document, measurement supplied, samples))))
        selected = U.Scope (U.ScopeId identity) document (measurement supplied) samples
        groups = Map.elems (Map.fromListWith (<>) [(U.unitId sample, sample :| []) | sample <- matched])
    -- Group directly from the checked nonempty inventory; Map traversal makes
    -- identity and weighting independent of artifact delivery order.
    case NonEmpty.nonEmpty (map aggregate groups) of
        Nothing -> Left EmptyDomain
        Just values -> pure (U.Observed selected values)

uniqueCases :: [Case] -> Either ObservationError ()
uniqueCases = visit Set.empty
  where
    visit _ [] = Right ()
    visit seen (item : remaining)
        | Set.member (caseKey item) seen = Left (DuplicateCase (caseKey item))
        | otherwise = visit (Set.insert (caseKey item) seen) remaining

validateDomain :: Domain.Domain -> Either ObservationError ()
validateDomain declared = do
    unless (all meaningful [Domain.domainName declared, Domain.unitDefinition declared]) (Left (InvalidDomain "domain and unit definitions must be named"))
    visit Set.empty (toList (Domain.declaredInputs declared))
  where
    meaningful = not . all isSpace
    visit _ [] = Right ()
    visit seen (item : remaining) = do
        unless (Set.notMember (Domain.inputKey item) seen) (Left (DuplicateInput (Domain.inputKey item)))
        unless (meaningful (Domain.unitId item)) (Left (InvalidDomain "empty unit identity"))
        unless (Domain.tokens item > 0 && Domain.temperature item > 0 && not (isInfinite (Domain.temperature item)) && '\0' `notElem` Domain.prompt item) (Left (InvalidDomain "invalid inference input declaration"))
        visit (Set.insert (Domain.inputKey item) seen) remaining

checked :: Maybe Measurement.Method -> (U.Key, (Domain.Input, (Numerical.BoundRun, [Numerical.Run]))) -> Either ObservationError U.Sample
checked selected (key, (input, (pair, repeated))) = do
    observed <- first (InvalidPair key) (Numerical.observe pair)
    before <- response key N.Reference (Numerical.reference pair)
    after <- response key N.Candidate (Numerical.candidate pair)
    let requested = Result.consumed before
        same label condition = unless condition (Left (InputMismatch key label))
    same "prompt" (Infer.prompt requested == Domain.prompt input)
    same "token budget" (Infer.tokens requested == Domain.tokens input)
    same "temperature" (Infer.temperature requested == Domain.temperature input)
    same "seed" (Infer.seed requested == Domain.seed input)
    let chain = Numerical.candidate pair : repeated
    invariance <- traverse adjacent (zip3 [0 ..] chain (drop 1 chain))
    reference <- traverse (measure N.Reference before) selected
    candidate <- traverse (measure N.Candidate after) selected
    pure (U.Sample key (Domain.unitId input) observed invariance reference candidate)
  where
    measure side result method = first (MeasurementFailed key side) (Measurement.observe method input result)
    adjacent (index, previous, next) = first (InvalidRepeat key index) (Numerical.observe (Numerical.BoundRun previous next))

response :: U.Key -> N.Side -> Numerical.Run -> Either ObservationError Result.Result
response key side run = Inference.result <$> first (InvalidResponse key side) (Inference.admit (Numerical.planned run) (Numerical.binding run) (Numerical.logBytes run))

consistent :: N.Side -> NonEmpty U.Sample -> Either ObservationError ()
consistent side (initial :| remaining) = visit Set.empty (initial : remaining)
  where
    expected = N.sourcePolicy (U.source side initial)
    visit _ [] = Right ()
    visit seen (sample : rest) = foldM (record (U.key sample)) seen (executions sample) >>= (`visit` rest)
    executions sample = U.source side sample : [next | side == N.Candidate, observed <- U.invariance sample, let N.Scope _ _ next _ _ _ = Numerical.scope observed]
    record key seen actual = do
        let binding = N.sourceBinding actual
            identity = (Invocation.boundCall binding, Invocation.boundAttempt binding, Invocation.boundInstance binding)
        unless (N.sourcePolicy actual == expected) (Left (MixedImplementation side key))
        unless (Set.notMember identity seen) (Left (ReusedExecution side key))
        pure (Set.insert identity seen)

aggregate :: NonEmpty U.Sample -> U.Unit
aggregate samples = U.Unit (U.unitId (NonEmpty.head samples)) (fmap U.key samples) (average U.reference) (average U.candidate)
  where
    average select = do
        measured <- traverse select samples
        pure (sum (fmap Measurement.value measured) / fromIntegral (length samples))
