{-# LANGUAGE OverloadedStrings #-}

module Statistics (statistics) where

import Hedgehog
import Invar.Use.Statistics qualified as S

statistics :: Group
statistics = Group "Paired statistics beside admission" [("normal quantile encloses known values from above", withTests 1 (property quantiles)), ("exact McNemar tail counts discordant units", withTests 1 (property mcnemar)), ("Wald interval, equivalence and bounds on known pairs", withTests 1 (property wald)), ("invalid alphas and samples are rejected", withTests 1 (property rejected))]

quantiles :: PropertyT IO ()
quantiles = mapM_ encloses ([(1 / 40, 1.959963984540054), (1 / 80, 2.241402727604947), (1 / 20, 1.6448536269514722)] :: [(Rational, Rational)])
  where
    encloses (alpha, known) = do
        upper <- evalMaybe (S.normalQuantileUpper alpha)
        assert (upper >= known - 1e-15 && upper - known < 1e-12)

mcnemar :: PropertyT IO ()
mcnemar = do
    balanced <- evalMaybe (S.compare (1 / 40) (1 / 40) (replicate 21 (0, 1) ++ replicate 20 (1, 0) ++ replicate 100 (0, 0)))
    S.mcnemar balanced === Just (S.McNemar 21 20 (1 / 2))
    lopsided <- evalMaybe (S.compare (1 / 40) (1 / 40) (replicate 3 (0, 1) ++ replicate 5 (1, 1)))
    S.mcnemar lopsided === Just (S.McNemar 3 0 (1 / 8))
    fractional <- evalMaybe (S.compare (1 / 40) (1 / 40) [(0, 1 / 2), (1, 1), (0, 0)])
    S.mcnemar fractional === Nothing

wald :: PropertyT IO ()
wald = do
    result <- evalMaybe (S.compare (1 / 40) 1 [(0, 0), (0, 1), (1, 0), (0, 0)])
    z <- evalMaybe (S.normalQuantileUpper (1 / 40))
    S.meanIncrease result === 0
    S.waldUpper result === negate (S.waldLower result)
    assert (abs (S.waldUpper result - z * 0.40824829046386301636) < 1e-12)
    assert (S.equivalent result && S.noninferior result)
    assert (S.bernsteinUpper result > S.waldUpper result)
    narrow <- evalMaybe (S.compare (1 / 40) (1 / 10) [(0, 0), (0, 1), (1, 0), (0, 0)])
    assert (not (S.equivalent narrow) && not (S.noninferior narrow))

rejected :: PropertyT IO ()
rejected = do
    S.normalQuantileUpper 0 === Nothing
    S.normalQuantileUpper (1 / 2) === Nothing
    S.compare (1 / 40) (1 / 40) [(0, 1)] === Nothing
    S.compare (1 / 2) (1 / 40) [(0, 1), (1, 0)] === Nothing
