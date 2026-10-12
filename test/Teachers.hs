{-# LANGUAGE OverloadedStrings #-}

module Teachers (teachers) where

import Calls (change, request, wire)
import Control.Monad (void)
import Data.Aeson (Value (..), eitherDecodeStrict, toJSON)
import Data.Aeson.KeyMap qualified as Fields
import Data.ByteString.Char8 qualified as Bytes
import Data.Word (Word32)
import Hedgehog
import InferenceObservations qualified as Source
import Invar.Infer qualified as Infer
import Invar.Infer.Observation qualified as Inference
import Invar.Learn qualified as L
import Invar.Learn.Algorithm qualified as Algorithm
import Invar.Policy qualified as Policy
import Invar.Score qualified as Score
import Invar.Score.Resident qualified as Hosted
import Invar.Spec.Invocation qualified as V
import Invar.Teacher qualified as Teacher
import Scores qualified

teachers :: Group
teachers =
    Group
        "Teacher scoring targets"
        [ ("a teacher declaration scores the student's path under the teacher's materialization, bound to the teacher's model description", once scored)
        , ("a teacher with another temperature cannot score the path", once refused)
        , ("preparing a teacher's score applies the teacher rule to its declaration", once ruled)
        , ("a teacher reading admits only finite nonpositive teacher words and keeps its score's call", once supported)
        , ("the joined result of a distillation sample covers its trajectory, the teacher's call and every teacher word", once joined)
        , ("a resident teacher loads its model before its first score and never again", once resident)
        ]
  where
    once = withTests 1 . property

teacher :: Algorithm.Teacher
teacher = Algorithm.Teacher "teacher-model" "teacher-revision" (replicate 64 '6') (Infer.tokenizer request) (replicate 64 '7') (replicate 64 '8') (Infer.temperature request) "none" "processed" "forced"

source :: PropertyT IO Inference.Source
source = do
    original <- Source.fixture
    planned <- evalEither (Infer.prepare request)
    Inference.source <$> evalEither (Source.admitted planned Source.bound (wire original))

scored :: PropertyT IO ()
scored = do
    path <- source
    plan <- evalEither (Teacher.prepare teacher path)
    let target = Score.target plan
        requested = Infer.requested target
    (Infer.artifact requested, Infer.tokenizer requested, Infer.base requested, Infer.assembly requested) === (replicate 64 '6', Infer.tokenizer request, replicate 64 '7', replicate 64 '8')
    (Infer.prompt requested, Infer.tokens requested, Infer.seed requested) === (Infer.prompt request, Infer.tokens request, Infer.seed request)
    description <- evalEither (Policy.describe ("teacher-model", "teacher-revision") (replicate 64 '6', Infer.tokenizer request, replicate 64 '7', replicate 64 '8'))
    Infer.boundPolicy target === Just description
    call <- evalEither (Score.bind (V.Binding (V.CallId 30) (V.AttemptId 31) (V.Instance 32)) plan)
    envelope <- evalEither (eitherDecodeStrict (Score.input call))
    case envelope of
        Object fields -> Fields.lookup "adapter" fields === Just (String "6666666666666666666666666666666666666666666666666666666666666666")
        _ -> failure

refused :: PropertyT IO ()
refused = do
    path <- source
    void (Teacher.prepare teacher {Algorithm.teacherTemperature = 1.1} path) === Left "The teacher scores at the temperature the student sampled at"

ruled :: PropertyT IO ()
ruled = do
    path <- source
    let declared = teacher {Algorithm.teacherLogprobs = "raw"}
    expected <- either pure (const failure) (Algorithm.teacher (Infer.tokenizer request) declared)
    void (Teacher.prepare declared path) === Left expected

supported :: PropertyT IO ()
supported = do
    original <- Source.fixture
    (call, events) <- Scores.fixtureFor original request
    let report encoded = evalEither (Score.admit call 0 (wire (Scores.changeObservation (change "log_probability_bits" (toJSON (encoded :: [Word32]))) events)))
    finite <- report [0xbf000000, 0xbe800000]
    Teacher.reading finite === Right (L.Reading (Score.binding call) [0xbf000000, 0xbe800000])
    unsupported <- report [0xbf000000, 0xff800000]
    Teacher.reading unsupported === Left "Every teacher word must be a finite nonpositive FP32 log probability"

joined :: PropertyT IO ()
joined = do
    let reading = L.Reading (V.Binding (V.CallId 30) (V.AttemptId 31) (V.Instance 32)) [0xbf000000, 0xbe800000]
        trajectory = replicate 64 'a'
        result = Teacher.joined trajectory reading
    assert (result `notElem` [Teacher.joined (replicate 64 'b') reading, Teacher.joined trajectory reading {L.readingCall = V.Binding (V.CallId 30) (V.AttemptId 31) (V.Instance 33)}, Teacher.joined trajectory reading {L.readingWords = [0xbf000000, 0xbe800001]}, Teacher.joined trajectory reading {L.readingWords = [0xbf000000]}])

resident :: PropertyT IO ()
resident = do
    let segment names = Bytes.unlines [Bytes.pack ("{\"stage\":\"" ++ name ++ "\"}") | name <- names ++ ["loaded_adapter", "consumed"]]
        accepted initial names = either (const False) (const True) (Hosted.readiness initial (segment names))
    [accepted True names | names <- [["load"], ["loading", "profile", "load"]]] === [True, True]
    [accepted True names | names <- [[], ["profile", "load"], ["load", "load"]]] === [False, False, False]
    [accepted False names | names <- [[], ["load"], ["loading", "profile", "load"]]] === [True, False, False]
