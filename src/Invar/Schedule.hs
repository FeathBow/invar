{-# LANGUAGE Safe #-}

module Invar.Schedule (Plan, Axis (..), Error (..), prepare, execute, deliver) where

import Control.Monad (unless)
import Data.List (sort)
import Data.Map.Strict qualified as Map
import Numeric.Natural (Natural)

data Axis = Execution | Delivery
    deriving (Eq, Show)

data Error = InvalidPermutation Axis | MemberMismatch
    deriving (Eq, Show)

data Plan = Plan [Natural] [Natural]

prepare :: Natural -> [Natural] -> [Natural] -> Either Error Plan
prepare count execution delivery = do
    let positions = takeWhile (< count) [0 ..]
    unless (sort execution == positions) (Left (InvalidPermutation Execution))
    unless (sort delivery == positions) (Left (InvalidPermutation Delivery))
    pure (Plan execution delivery)

execute :: Plan -> [value] -> Either Error [(Natural, value)]
execute (Plan order _) = select order . zip [0 ..]

deliver :: Plan -> [(Natural, value)] -> Either Error [(Natural, value)]
deliver (Plan _ order) = select order

select :: [Natural] -> [(Natural, value)] -> Either Error [(Natural, value)]
select order values = do
    unless (sort (map fst values) == sort order) (Left MemberMismatch)
    let indexed = Map.fromList values
    traverse (\position -> maybe (Left MemberMismatch) (Right . (position,)) (Map.lookup position indexed)) order
