{-# LANGUAGE Safe #-}

module Invar.Use.Execution (Order (..), Arrangement (..), arrange, unchanged) where

import Data.List (genericDrop, genericLength, genericSplitAt, genericTake)
import Numeric.Natural (Natural)

data Order = Declared | Reversed | Rotated Natural
    deriving (Eq, Show)

data Arrangement = Arrangement {order :: Order, groupSize :: Natural}
    deriving (Eq, Show)

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
