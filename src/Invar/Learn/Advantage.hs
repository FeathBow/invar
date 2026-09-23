module Invar.Learn.Advantage (Reward (..), Error (..), calculate, sum64) where

import Control.Monad (foldM_, unless)
import Data.List (sortOn)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Word (Word32)
import GHC.Float (castFloatToWord32, double2Float)

data Reward = Reward {sample :: String, group :: String, value :: Double}
    deriving (Eq, Show)

data Error = InvalidInput String | NonFinite String
    deriving (Eq, Show)

calculate :: Double -> [Reward] -> Either Error (Map String Word32)
calculate delta supplied = do
    _ <- finite "advantage delta" delta
    unless (delta > 0) (Left (InvalidInput "Advantage delta must be positive"))
    unless (not (null supplied) && Set.size names == length supplied) (Left (InvalidInput "Rewards must have distinct sample identities"))
    mapM_ validate supplied
    Map.fromList . concat <$> traverse (normalize delta . sortOn sample) (Map.elems grouped)
  where
    names = Set.fromList (map sample supplied)
    grouped = Map.fromListWith (++) [(group item, [item]) | item <- supplied]
    validate item = do
        unless (not (null (sample item)) && not (null (group item))) (Left (InvalidInput "Reward sample and group identities must be nonempty"))
        _ <- finite "reward" (value item)
        pure ()

normalize :: Double -> [Reward] -> Either Error [(String, Word32)]
normalize delta members = do
    unless (length members >= minimumGroup) (Left (InvalidInput "Each reward group must contain at least two samples"))
    total <- sum64 (map value members)
    mean <- finite "reward mean" (total / count)
    centered <- traverse (finite "centered reward" . subtract mean . value) members
    squared <- traverse (\deviation -> finite "squared deviation" (deviation * deviation)) centered
    squares <- sum64 squared
    variance <- finite "population variance" (squares / count)
    deviation <- finite "population standard deviation" (sqrt variance)
    denominator <- finite "advantage denominator" (deviation + delta)
    normalized <- traverse (\numerator -> finite "advantage" (numerator / denominator) >>= word) centered
    pure (zip (map sample members) normalized)
  where
    count = fromIntegral (length members)
    minimumGroup = 2

word :: Double -> Either Error Word32
word number =
    let rounded = double2Float number
     in if isNaN rounded || isInfinite rounded
            then Left (NonFinite "FP32 advantage")
            else Right (castFloatToWord32 rounded)

sum64 :: [Double] -> Either Error Double
sum64 values = do
    operands <- traverse (finite "summation operand") values
    foldM_ accumulate [] operands
    finite "exact sum" (fromRational (sum (map toRational operands)))
  where
    accumulate partials operand = merge operand [] partials
    merge operand kept [] = Right (reverse kept ++ [operand | operand /= 0])
    merge operand kept (partial : remaining) = do
        let (large, small) = if abs operand < abs partial then (partial, operand) else (operand, partial)
        high <- finite "summation intermediate" (large + small)
        let low = small - (high - large)
        merge high (if low == 0 then kept else low : kept) remaining

finite :: String -> Double -> Either Error Double
finite stage number
    | isNaN number || isInfinite number = Left (NonFinite stage)
    | otherwise = Right number
