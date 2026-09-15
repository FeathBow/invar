{-# LANGUAGE OverloadedStrings #-}

module Invar.Quality.Summary (Sample (..), compare, measure) where

import Control.Monad (unless)
import Data.Aeson (Value, object, (.=))
import Data.Aeson.Types qualified as Json
import Data.List.NonEmpty (NonEmpty (..))
import Data.List.NonEmpty qualified as NonEmpty
import Data.Map.Strict qualified as Map
import Data.Ratio ((%))
import Numeric.Natural (Natural)
import Prelude hiding (compare)

-- Numerical summary inputs carry no execution or invocation authority.
data Sample = Sample
    { sampleCohort :: Natural
    , sampleName :: String
    , sampleGroup :: String
    , sampleSeed :: Integer
    , sampleReward :: Natural
    , sampleTokens :: Natural
    , sampleTruncated :: Bool
    }

type Pair = (Sample, Sample)

data Metrics = Metrics {sampleCount :: Natural, rewardSum :: Natural, responseTokens :: Natural, truncatedCount :: Natural}

data Summary = Summary {initial :: Metrics, trained :: Metrics, improved :: Natural, worsened :: Natural, unchanged :: Natural}

compare :: [Sample] -> [Sample] -> Either String [Json.Pair]
compare first second = do
    pairs <- paired first second
    overall <- summaryValue (summarize pairs)
    seeds <- traverse (summaryValue . summarize) (grouped sampleSeed pairs)
    let groups = Map.map summarize (grouped (\sample -> (sampleCohort sample, sampleGroup sample)) pairs)
    observedGroups <- traverse summaryValue groups
    pure ["overall" .= overall, "group_count" .= Map.size groups, "zero_variance_groups" .= object ["initial" .= constantGroups initial groups, "trained" .= constantGroups trained groups], "by_seed" .= [object ["seed" .= seed, "comparison" .= result] | (seed, result) <- Map.toAscList seeds], "by_group" .= [object ["cohort" .= cohort, "group" .= group, "comparison" .= result] | ((cohort, group), result) <- Map.toAscList observedGroups]]

paired :: [Sample] -> [Sample] -> Either String (NonEmpty Pair)
paired first second = do
    left <- indexed first
    right <- indexed second
    unless (Map.keysSet left == Map.keysSet right) (Left "Compared evaluations must have the same sample inventory")
    pairs <- maybe (Left "Missing paired evaluation samples") Right (NonEmpty.nonEmpty (Map.elems (Map.intersectionWith (,) left right)))
    unless (all (\(before, after) -> sampleSeed before == sampleSeed after && sampleGroup before == sampleGroup after) pairs) (Left "Compared sample groups or seeds differ")
    pure pairs

measure :: [Sample] -> Either String Value
measure observed = do
    _ <- indexed observed
    values <- maybe (Left "Missing evaluation samples") Right (NonEmpty.nonEmpty observed)
    metricsValue (metrics values)

indexed :: [Sample] -> Either String (Map.Map (Natural, String) Sample)
indexed observed = do
    let values = Map.fromList [((sampleCohort sample, sampleName sample), sample) | sample <- observed]
    unless (Map.size values == length observed) (Left "Repeated quality summary sample")
    pure values

grouped :: (Ord key) => (Sample -> key) -> NonEmpty Pair -> Map.Map key (NonEmpty Pair)
grouped key pairs = Map.fromListWith (<>) [(key (fst pair), pair :| []) | pair <- NonEmpty.toList pairs]

metrics :: NonEmpty Sample -> Metrics
metrics samples = Metrics (fromIntegral (NonEmpty.length samples)) (sum (fmap sampleReward samples)) (sum (fmap sampleTokens samples)) (fromIntegral (length (filter sampleTruncated (NonEmpty.toList samples))))

summarize :: NonEmpty Pair -> Summary
summarize pairs = Summary (metrics (fmap fst pairs)) (metrics (fmap snd pairs)) (count (<)) (count (>)) (count (==))
  where
    count relation = fromIntegral (length [() | (left, right) <- NonEmpty.toList pairs, relation (sampleReward left) (sampleReward right)])

constantGroups :: (Summary -> Metrics) -> Map.Map key Summary -> Int
constantGroups side = length . filter constant . Map.elems
  where
    constant summary = let measured = side summary in rewardSum measured == 0 || rewardSum measured == sampleCount measured

metricsValue :: Metrics -> Either String Value
metricsValue values = do
    meanReward <- ratio (toInteger (rewardSum values)) (sampleCount values)
    meanTokens <- ratio (toInteger (responseTokens values)) (sampleCount values)
    truncation <- ratio (toInteger (truncatedCount values)) (sampleCount values)
    pure (object ["sample_count" .= sampleCount values, "reward_sum" .= rewardSum values, "response_tokens" .= responseTokens values, "truncated_count" .= truncatedCount values, "reward_mean" .= meanReward, "mean_response_tokens" .= meanTokens, "truncation_rate" .= truncation])

summaryValue :: Summary -> Either String Value
summaryValue values = do
    before <- metricsValue (initial values)
    after <- metricsValue (trained values)
    let change = toInteger (rewardSum (trained values)) - toInteger (rewardSum (initial values))
    meanChange <- ratio change (sampleCount (initial values))
    pure (object ["initial" .= before, "trained" .= after, "reward_sum_change" .= change, "reward_mean_change" .= meanChange, "improved_samples" .= improved values, "worsened_samples" .= worsened values, "unchanged_samples" .= unchanged values])

ratio :: Integer -> Natural -> Either String Double
ratio numerator denominator =
    let value = fromRational (numerator % toInteger denominator)
     in if isNaN value || isInfinite value then Left "Non-finite evaluation summary ratio" else Right value
