{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE Safe #-}

module Invar.Use.Prepare (UnitReport (..), Sampling (..), Shape (..), shape, expectedPremises, premisesFor, units, basisBytes) where

import Data.ByteString (ByteString)
import Data.Char (ord)
import Data.Foldable (toList)
import Data.List (intercalate, nub, sort, sortOn)
import Data.Map.Strict qualified as Map
import Data.Text qualified as Text
import Data.Text.Encoding qualified as Text
import Invar.Spec.Domain qualified as Domain
import Invar.Spec.Measurement qualified as Measurement
import Invar.Use.Contract qualified as C
import Numeric (showHex)
import Numeric.Natural (Natural)

data UnitReport = UnitReport {inputCount :: Natural, unitCount :: Natural, seedsPerUnit :: [(Natural, Natural)], candidateExecutions :: Natural}
    deriving (Eq, Show)

data Sampling = Finite | Hoeffding | Bernstein | Conditional
    deriving (Eq, Show)

data Shape = Shape {measuredLoss :: Maybe Measurement.Method, sampling :: Maybe Sampling, numerical :: Bool, invariant :: Bool}

shape :: C.UseContract -> Shape
shape contract =
    Shape
        (C.lossRequirement criterion *> C.declaredMeasurement contract)
        (kind . C.standard <$> C.lossRequirement criterion)
        (not (null (C.numericalRequirements criterion)))
        (not (null (C.invarianceRequirements criterion)))
  where
    criterion = C.criterion contract
    kind C.FiniteDomain = Finite
    kind (C.HoeffdingPopulation _) = Hoeffding
    kind (C.EmpiricalBernsteinPopulation _) = Bernstein
    kind (C.ConditionalDerivation _) = Conditional

expectedPremises :: C.UseContract -> [C.Premise]
expectedPremises = premisesFor . shape

premisesFor :: Shape -> [C.Premise]
premisesFor declared = sort (nub (admission ++ measured ++ generated ++ scheduled ++ sampled))
  where
    admission = [C.RequirementsJustified, C.ContractFrozen, C.AcceptanceIsolation, C.SelectionControl]
    measured = case measuredLoss declared of
        Just method ->
            let bindings = Map.elems (Measurement.bindings (Measurement.specification method))
             in [C.MeasurementMeaning, C.ExecutionAuthenticity]
                    ++ [C.MeasurementInput | any observed bindings]
                    ++ [C.ParameterMeaning | any parameter bindings]
        Nothing -> []
    generated = [premise | numerical declared || invariant declared, premise <- [C.ExecutionAuthenticity, C.SelectedBehavior, C.OwnCacheExecution]]
    scheduled = [C.ScheduleVariation | invariant declared]
    sampled = case sampling declared of
        Just Hoeffding -> [C.IndependentUnits, C.ReplicateSamplingLaw]
        Just Bernstein -> [C.BernsteinIndependentUnits, C.BernsteinIdenticalUnits, C.BernsteinReplicateSamplingLaw]
        _ -> []
    observed (Measurement.ObservedField _) = True
    observed _ = False
    parameter (Measurement.Parameter _) = True
    parameter _ = False

units :: C.UseContract -> UnitReport
units contract = UnitReport (count inputs) (count grouped) (Map.toList sizes) executions
  where
    inputs = toList (Domain.declaredInputs (C.declaredDomain contract))
    grouped = Map.fromListWith (+) [(Domain.unitId input, 1 :: Natural) | input <- inputs]
    sizes = Map.fromListWith (+) [(size, 1) | size <- Map.elems grouped]
    executions = maximum (1 : map C.executions (C.invarianceRequirements (C.criterion contract)))
    count :: (Foldable container) => container value -> Natural
    count = fromIntegral . length

basisBytes :: [(String, String)] -> Either String ByteString
basisBytes files
    | null files = Left "A basis needs at least one file"
    | not (all (logical . fst) files) = Left "Basis paths are relative to the declaration, use / separators and contain no . or .. segment"
    | length (nub (map fst files)) /= length files = Left "A basis names the same file twice"
    | otherwise = Right (Text.encodeUtf8 (Text.pack ("[" ++ commas ["[" ++ quoted name ++ "," ++ quoted digest ++ "]" | (name, digest) <- sortOn fst files] ++ "]")))
  where
    logical path = not (null path) && take 1 path /= "/" && all segment (split path) && notElem '\\' path
    segment part = part `notElem` ["", ".", ".."]
    commas = intercalate ","
    quoted value = "\"" ++ concatMap escape value ++ "\""
    escape character = case character of
        '"' -> "\\\""
        '\\' -> "\\\\"
        '\b' -> "\\b"
        '\f' -> "\\f"
        '\n' -> "\\n"
        '\r' -> "\\r"
        '\t' -> "\\t"
        _ | ord character < 32 -> "\\u" ++ replicate (4 - length (showHex (ord character) "")) '0' ++ showHex (ord character) ""
        _ -> [character]
    split path = case break (== '/') path of
        (part, []) -> [part]
        (part, _ : rest) -> part : split rest
