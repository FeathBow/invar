{-# LANGUAGE Safe #-}

module Invar.Use.Execution (
    Order (..),
    Arrangement (..),
    arrange,
    unchanged,
    Scorer (..),
    ProbePlan (..),
    Collection (..),
    emptyCollection,
    CollectionProblem (..),
    collectionProblems,
) where

import Data.List (genericDrop, genericLength, genericSplitAt, genericTake, nub)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Invar.Spec.Numerical qualified as N
import Invar.Use.Contract qualified as C
import Numeric.Natural (Natural)

data Order = Declared | Reversed | Rotated Natural
    deriving (Eq, Show)

data Arrangement = Arrangement {order :: Order, groupSize :: Natural}
    deriving (Eq, Show)

data Scorer = Scorer
    { scorerBackend :: String
    , scorerWorker :: FilePath
    , scorerConfiguration :: FilePath
    , scorerAdapter :: FilePath
    }
    deriving (Eq, Show)

data ProbePlan = ProbePlan {probePath :: N.Side, probeSteps :: [Natural]}
    deriving (Eq, Show)

data Collection = Collection
    { scorePaths :: [N.Side]
    , probes :: [ProbePlan]
    , scorers :: Map N.Side Scorer
    }
    deriving (Eq, Show)

emptyCollection :: Collection
emptyCollection = Collection [] [] Map.empty

data CollectionProblem = CollectionProblem {collectionAt :: String, collectionMessage :: String}
    deriving (Eq, Show)

collectionProblems :: C.UseContract -> Collection -> [CollectionProblem]
collectionProblems contract collection = scoreProblems ++ probeProblems ++ scorerProblems
  where
    requirements = C.numericalRequirements (C.criterion contract)
    requiredScores = nub [side | requirement <- requirements, N.ScoredPathLogRatioWithin side _ <- [C.relation requirement]]
    requiredProbes = requiredProbePlans requirements
    scoreProblems =
        duplicates "execution:collection.scores" (scorePaths collection)
            ++ [CollectionProblem "execution:collection.scores" ("The contract does not require a score of the " ++ sideName side ++ " path") | side <- scorePaths collection, side `notElem` requiredScores]
            ++ [CollectionProblem "execution:collection.scores" ("The contract requires a score of the " ++ sideName side ++ " path") | side <- requiredScores, side `notElem` scorePaths collection]
    probeProblems =
        duplicates "execution:collection.probes" (map probePath (probes collection))
            ++ [CollectionProblem "execution:collection.probes" ("The contract does not require a probe of the " ++ sideName side ++ " path") | side <- map probePath (probes collection), side `notElem` map fst requiredProbes]
            ++ [CollectionProblem ("execution:collection.probes[" ++ show index ++ "]") ("Probe steps for the " ++ sideName (probePath plan) ++ " path must " ++ expectation steps (probeSteps plan)) | (index, (plan, steps)) <- zip [0 :: Int ..] [(plan, steps) | plan <- probes collection, Just steps <- [lookup (probePath plan) requiredProbes]], probeSteps plan /= steps]
            ++ [CollectionProblem "execution:collection.probes" ("The contract requires a probe of the " ++ sideName side ++ " path") | (side, _) <- requiredProbes, side `notElem` map probePath (probes collection)]
            ++ [CollectionProblem "execution:collection.probes" ("The contract names conflicting probe steps for the " ++ sideName side ++ " path") | side <- nub (map fst requiredProbes), length [steps | (selected, steps) <- requiredProbes, selected == side] > 1]
    scorerProblems =
        [ CollectionProblem ("execution:collection.scorers." ++ sideName side) ("The plan has no scorer for the " ++ sideName side ++ " implementation")
        | side <- nub (scoreTargets ++ probeTargets)
        , Map.notMember side (scorers collection)
        ]
            ++ [CollectionProblem ("execution:collection.scorers." ++ sideName side) "A probe target must use the mlx or vllm backend" | not (null (probes collection)), side <- [N.Reference, N.Candidate], Just scorer <- [Map.lookup side (scorers collection)], scorerBackend scorer `notElem` ["mlx", "vllm"]]
            ++ [CollectionProblem ("execution:collection.scorers." ++ sideName side) "The Hugging Face scorer records no full-vocabulary probe" | not (null (probes collection)), side <- [N.Reference, N.Candidate], Just scorer <- [Map.lookup side (scorers collection)], scorerBackend scorer == "hf"]
    scoreTargets = [otherSide side | side <- scorePaths collection]
    probeTargets = [side | _ <- probes collection, side <- [N.Reference, N.Candidate]]
    requiredProbePlans selected = [(side, steps) | requirement <- selected, let steps = C.probeSteps requirement, N.FullVocabularyKLWithin side _ _ <- [C.relation requirement]]
    duplicates at values =
        [CollectionProblem at ("The plan names the " ++ sideName side ++ " path more than once") | side <- nub values, length (filter (== side) values) > 1]
    expectation expected actual = "match the contract (" ++ show expected ++ "), not " ++ show actual

sideName :: N.Side -> String
sideName N.Reference = "reference"
sideName N.Candidate = "candidate"

otherSide :: N.Side -> N.Side
otherSide N.Reference = N.Candidate
otherSide N.Candidate = N.Reference

arrange :: Arrangement -> [value] -> [[value]]
arrange (Arrangement selected size) values
    | size == 0 = []
    | otherwise = chunks (ordered selected)
  where
    ordered Declared = values
    ordered Reversed = reverse values
    ordered (Rotated offset) =
        let shift = if null values then 0 else offset `mod` genericLength values
         in genericDrop shift values ++ genericTake shift values
    chunks [] = []
    chunks remaining = let (group, rest) = genericSplitAt size remaining in group : chunks rest

unchanged :: Natural -> Arrangement -> [Arrangement] -> [Natural]
unchanged count paired repeats =
    [ position
    | (position, arrangement) <- zip [0 ..] repeats
    , let schedule = arrange arrangement indices
    , schedule `elem` (arrange paired indices : [arrange other indices | (index, other) <- zip [0 :: Natural ..] repeats, index /= position])
    ]
  where
    indices = genericTake count [0 ..] :: [Natural]
