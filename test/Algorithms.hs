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
        , ("the teacher rule accepts a complete declaration with the student's tokenizer and refuses each field outside it", once teaching)
        , ("the settings check refuses a distillation teacher and coefficients through these rules", once taughtSettings)
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

teacher :: Algorithm.Teacher
teacher = Algorithm.Teacher "teacher-model" "teacher-revision" (replicate 64 'c') (replicate 64 '7') (replicate 64 '8') (replicate 64 '9') 0.8 "none" "processed" "forced"

teaching :: PropertyT IO ()
teaching = do
    Algorithm.teacher (replicate 64 'c') teacher === Right ()
    let refused =
            [ (teacher {Algorithm.teacherModel = ""}, "The teacher's model and revision must be nonempty")
            , (teacher {Algorithm.teacherRevision = ""}, "The teacher's model and revision must be nonempty")
            , (teacher {Algorithm.teacherProfile = "profile"}, "Expected lowercase SHA-256 teacher identities")
            , (teacher {Algorithm.teacherBase = replicate 64 'A'}, "Expected lowercase SHA-256 teacher identities")
            , (teacher {Algorithm.teacherTokenizer = replicate 64 'd'}, "The teacher's tokenizer must be the student's")
            , (teacher {Algorithm.teacherTemperature = 0}, "The teacher's temperature must be positive")
            , (teacher {Algorithm.teacherTemperature = nan}, "The teacher's temperature must be positive")
            , (teacher {Algorithm.teacherTemperature = infinity}, "The teacher's temperature must be positive")
            , (teacher {Algorithm.teacherTruncation = "top-k"}, "The teacher must score the forced path with processed log probabilities and no truncation")
            , (teacher {Algorithm.teacherLogprobs = "raw"}, "The teacher must score the forced path with processed log probabilities and no truncation")
            , (teacher {Algorithm.teacherPath = "prompt"}, "The teacher must score the forced path with processed log probabilities and no truncation")
            ]
    [Algorithm.teacher (replicate 64 'c') chosen | (chosen, _) <- refused] === [Left message | (_, message) <- refused]

taughtSettings :: PropertyT IO ()
taughtSettings = do
    let taught = F.configured {L.algorithm = L.Opd teacher {Algorithm.teacherTokenizer = L.tokenizer F.configured}}
    L.validate taught === Right ()
    L.validate taught {L.algorithm = L.Opd teacher {Algorithm.teacherTokenizer = replicate 64 'd'}} === Left (L.InvalidSettings "The teacher's tokenizer must be the student's")
    L.validate taught {L.clip = 0} === Left (L.InvalidSettings "Invalid distillation coefficients")
