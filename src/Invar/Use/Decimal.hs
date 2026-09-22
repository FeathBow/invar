{-# LANGUAGE OverloadedStrings #-}

module Invar.Use.Decimal (Error (..), domain, method, bind) where

import Control.Monad (unless)
import Data.Aeson (encode)
import Data.ByteString.Char8 qualified as Bytes
import Data.ByteString.Lazy qualified as Lazy
import Data.List.NonEmpty qualified as NonEmpty
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Invar.Reward qualified as Reward
import Invar.Reward.Program qualified as RewardProgram
import Invar.Spec.Artifact qualified as Artifact
import Invar.Spec.Domain qualified as D
import Invar.Spec.Measurement qualified as M
import Invar.Spec.Program qualified as Program
import Invar.Spec.Value (Scalar (..), Value (..))
import Invar.Use.Observation qualified as Observation
import Invar.Workload qualified as Workload

data Error = EmptyWorkload | InconsistentQuestion String | RepeatedSeed String Integer | InvalidMethod M.Error
    deriving (Eq, Show)

-- This adapter owns the numeric answer format and prompt/seed grouping.
-- The generic observer and evidence rules have neither dependency.
domain :: Workload.Document -> Either Error D.Domain
domain document = do
    let tasks = [(D.Key cohort (Workload.name task), task) | (cohort, declaredCycle) <- zip [0 ..] (Workload.cycles document), task <- Workload.tasks declaredCycle]
    validate (map snd tasks)
    inputs <- maybe (Left EmptyWorkload) Right (NonEmpty.nonEmpty (map input tasks))
    pure (D.Domain "exact-decimal workload" (Bytes.pack (Workload.digest document) <> "\n" <> Lazy.toStrict (encode (Workload.value document))) "Exact prompt; equal declared seeds within prompt; equal prompts" inputs)
  where
    input (key, task) =
        D.Input
            key
            (Workload.prompt task)
            (Workload.prompt task)
            (Workload.tokens task)
            (Workload.temperature task)
            (Workload.seed task)
            (Map.singleton "expected" (Atom (Number (Reward.expected (Workload.rule task)))))

validate :: [Workload.Task] -> Either Error ()
validate = visit Map.empty Set.empty
  where
    visit _ _ [] = Right ()
    visit known trials (item : remaining) = do
        let prompt = Workload.prompt item
            trial = (prompt, Workload.seed item)
            protocol = (Workload.tokens item, Workload.temperature item, Workload.rule item)
        unless (maybe True (== protocol) (Map.lookup prompt known)) (Left (InconsistentQuestion prompt))
        unless (Set.notMember trial trials) (Left (RepeatedSeed prompt (Workload.seed item)))
        visit (Map.insert prompt protocol known) (Set.insert trial trials) remaining

method :: Either Error M.Method
method = do
    checked <- either (Left . InvalidMethod . M.InvalidProgram) Right RewardProgram.checked
    either (Left . InvalidMethod) Right (M.prepare (M.MethodSpec (Artifact.bytes checked) sources "score" "exact-decimal/v1" 0 1 M.DecreasingLoss "One minus exact-decimal reward; truncation is loss one"))
  where
    sources = Map.fromList [(Program.Semantic "rule", M.Parameter "expected"), (Program.Semantic "response", M.ObservedField M.Response), (Program.Semantic "truncated", M.ObservedField M.Truncated)]

bind :: Workload.Document -> [Observation.Case] -> Either Error Observation.BoundRun
bind workload cases = Observation.BoundRun <$> domain workload <*> (Just <$> method) <*> pure cases
