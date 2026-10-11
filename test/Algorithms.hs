{-# LANGUAGE OverloadedStrings #-}

module Algorithms (algorithms) where

import Hedgehog
import Invar.Learn qualified as L
import Invar.Learn.Algorithm qualified as Algorithm
import LearnerFixture qualified as F

algorithms :: Group
algorithms =
    Group
        "Learning algorithm definition"
        [ ("each GRPO coefficient rule accepts its range and refuses every value outside it, and the three hold only together", once grpo)
        , ("the AdamW rule accepts zero rate, decay and moments and refuses each scalar outside its range", once adamw)
        , ("the settings check refuses GRPO and AdamW coefficients through these rules", once settings)
        ]
  where
    once = withTests 1 . property

nan :: Double
nan = 0 / 0

infinity :: Double
infinity = 1 / 0

grpo :: PropertyT IO ()
grpo = do
    filter (not . Algorithm.clipping) [5e-324, 0.2, 0.9999999] === []
    filter Algorithm.clipping [0, 1, -0.1, 1.5, nan, infinity, -infinity] === []
    filter (not . Algorithm.penalty) [0, 0.04, 1e100] === []
    filter Algorithm.penalty [-5e-324, -1, nan, infinity] === []
    filter (not . Algorithm.delta) [5e-324, 0.0001, 1e100] === []
    filter Algorithm.delta [0, -1, nan, infinity] === []
    map (\(chosenClipping, chosenPenalty, chosenDelta) -> Algorithm.coefficients chosenClipping chosenPenalty chosenDelta) [(0.2, 0.04, 0.0001), (0, 0.04, 0.0001), (0.2, -1, 0.0001), (0.2, 0.04, 0)] === [True, False, False, False]

adamw :: PropertyT IO ()
adamw = do
    let valid = Algorithm.Optimizer 0.002 0.8 0.95 1e-7 0.01
        refused =
            [valid {Algorithm.learningRate = value} | value <- [-1, nan, infinity]]
                ++ [valid {Algorithm.firstMoment = value} | value <- [1, -0.1, nan]]
                ++ [valid {Algorithm.secondMoment = value} | value <- [1, infinity]]
                ++ [valid {Algorithm.epsilon = value} | value <- [0, -1, nan, infinity]]
                ++ [valid {Algorithm.weightDecay = value} | value <- [-1, nan, infinity]]
    filter (not . Algorithm.optimizer) [valid, Algorithm.Optimizer 0 0 0 1e-7 0] === []
    filter Algorithm.optimizer refused === []

settings :: PropertyT IO ()
settings = do
    L.validate F.configured === Right ()
    L.validate F.configured {L.clip = 0} === Left (L.InvalidSettings "Invalid GRPO coefficients")
    L.validate F.configured {L.optimizer = (L.optimizer F.configured) {L.epsilon = 0}} === Left (L.InvalidSettings "Invalid AdamW coefficients")
