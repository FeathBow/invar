{-# LANGUAGE LambdaCase #-}

module UseExecution (Execution (..), Side (..), decode, schedules, collectionCheck) where

import Check (Check, andThen, problem, value)
import Data.Aeson (Object, Value (..))
import Data.Foldable (toList)
import Data.Map.Strict qualified as Map
import Invar.Numerical qualified as N
import Invar.Use qualified as U
import Invar.Use.Execution qualified as E
import Located qualified as L
import Numeric.Natural (Natural)

data Side = Side {worker :: FilePath, configuration :: FilePath, adapter :: FilePath}

data Execution = Execution {backend :: String, python :: FilePath, cache :: FilePath, reference :: Side, candidate :: Side, paired :: E.Arrangement, repeats :: [E.Arrangement], collection :: E.Collection}

decode :: Value -> Check Execution
decode document =
    L.object top document `andThen` \fields ->
        L.only top ["format", "backend", "python", "cache", "sides", "paired", "repeats", "collection"] fields
            *> formatted fields
            *> ( Execution
                    <$> (text fields "backend" `andThen` supported)
                    <*> text fields "python"
                    <*> text fields "cache"
                    <*> side fields "reference"
                    <*> side fields "candidate"
                    <*> (L.field "invalid-value" top fields "paired" `andThen` arrangement (L.child top "paired"))
                    <*> (L.field "invalid-value" top fields "repeats" `andThen` L.list (L.child top "repeats") `andThen` traverse (\(index, entry) -> arrangement (L.item (L.child top "repeats") index) entry) . zip [0 ..])
                    <*> (L.field "invalid-value" top fields "collection" `andThen` collectionField)
               )
  where
    top = L.root "execution"
    text fields name = L.field "invalid-value" top fields name `andThen` L.text "invalid-value" (L.child top name)
    formatted fields = text fields "format" `andThen` \found -> if found == "invar-use-execution-v1" then value () else problem "invalid-value" (L.render (L.child top "format")) "Expected invar-use-execution-v1"
    supported found = if found `elem` ["mlx", "vllm"] then value found else problem "unsupported" (L.render (L.child top "backend")) "Only the mlx and vllm backends can run a contract"
    side fields name =
        let sides = L.child top "sides"
            path = L.child sides name
         in L.field "invalid-value" top fields "sides" `andThen` L.object sides `andThen` \entries ->
                L.field "invalid-value" sides entries name `andThen` L.object path `andThen` \entry ->
                    L.only path ["worker", "configuration", "adapter"] entry
                        *> (Side <$> named path entry "worker" <*> named path entry "configuration" <*> named path entry "adapter")
    named path entry name = L.field "invalid-value" path entry name `andThen` L.text "invalid-value" (L.child path name)
    collectionField found =
        let path = L.child top "collection"
         in L.object path found `andThen` \fields ->
                L.only path ["scores", "full_vocabulary_steps", "probes", "scorers"] fields
                    `andThen` \() ->
                        (case L.optionalField fields "scores" of Nothing -> value []; Just selected -> scores path selected)
                            `andThen` \scores' ->
                                (case L.optionalField fields "full_vocabulary_steps" of Nothing -> value (); Just selected -> legacySteps path selected)
                                    `andThen` \() ->
                                        (case L.optionalField fields "probes" of Nothing -> value []; Just value' -> L.list (L.child path "probes") value' `andThen` traverse (\(index, entry) -> probePlan (L.item (L.child path "probes") index) entry) . zip [0 ..])
                                            `andThen` \probes' ->
                                                (case L.optionalField fields "scorers" of Nothing -> value Map.empty; Just value' -> scorers path value')
                                                    `andThen` \scorers' -> value (E.Collection scores' probes' scorers')
    scores path value' = case value' of
        Bool False -> value []
        Bool True -> problem "unsupported" (L.render (L.child path "scores")) "Scored paths need structured collection entries and scorers"
        Array values -> traverse (\(index, entry) -> sideName (L.item (L.child path "scores") index) entry) (zip [0 ..] (toList values))
        _ -> problem "invalid-value" (L.render (L.child path "scores")) "Expected false or a list of reference/candidate paths"
    legacySteps path value' = case value' of
        Array values | null values -> value ()
        Array _ -> problem "unsupported" (L.render (L.child path "full_vocabulary_steps")) "Full-vocabulary probes need structured collection entries and scorers"
        _ -> problem "invalid-value" (L.render (L.child path "full_vocabulary_steps")) "Expected a list of probe steps"
    probePlan path found =
        L.object path found `andThen` \fields ->
            L.only path ["path", "steps"] fields
                *> (E.ProbePlan <$> (L.field "invalid-value" path fields "path" `andThen` sideName (L.child path "path")) <*> (L.field "invalid-value" path fields "steps" `andThen` steps (L.child path "steps")))
    steps path values = L.list path values `andThen` \entries -> traverse (\(index, entry) -> L.natural (L.item path index) entry) (zip [0 ..] entries)
    scorers path found =
        L.object path found `andThen` \fields ->
            L.only path ["reference", "candidate"] fields
                `andThen` \() ->
                    traverse (entry path fields) ["reference", "candidate"]
                        `andThen` \entries -> value (Map.fromList [(sideValue name, selectedScorer) | (name, Just selectedScorer) <- entries])
      where
        entry base selectedFields selectedName = case L.optionalField selectedFields selectedName of
            Nothing -> value (selectedName, Nothing)
            Just selectedValue -> (,) selectedName . Just <$> scorer (L.child base selectedName) selectedValue
    scorer path found =
        L.object path found `andThen` \fields ->
            L.only path ["backend", "worker", "configuration", "adapter"] fields
                *> (E.Scorer <$> (L.field "invalid-value" path fields "backend" `andThen` L.text "invalid-value" (L.child path "backend")) <*> (L.field "invalid-value" path fields "worker" `andThen` L.text "invalid-value" (L.child path "worker")) <*> (L.field "invalid-value" path fields "configuration" `andThen` L.text "invalid-value" (L.child path "configuration")) <*> (L.field "invalid-value" path fields "adapter" `andThen` L.text "invalid-value" (L.child path "adapter")))
    sideName path found =
        L.text "invalid-value" path found `andThen` \case
            "reference" -> value N.Reference
            "candidate" -> value N.Candidate
            _ -> problem "invalid-value" (L.render path) "Expected reference or candidate"
    sideValue name = if name == "reference" then N.Reference else N.Candidate

arrangement :: L.Path -> Value -> Check E.Arrangement
arrangement path found =
    L.object path found `andThen` \fields ->
        L.only path ["order", "offset", "group_size"] fields
            *> (E.Arrangement <$> order fields <*> (L.field "invalid-value" path fields "group_size" `andThen` L.positive (L.child path "group_size")))
  where
    order :: Object -> Check E.Order
    order fields =
        L.field "invalid-value" path fields "order" `andThen` L.text "invalid-value" (L.child path "order") `andThen` \name -> case (name, L.optionalField fields "offset") of
            ("declared", Nothing) -> value E.Declared
            ("reversed", Nothing) -> value E.Reversed
            ("rotated", Just offset) -> E.Rotated <$> L.natural (L.child path "offset") offset
            ("rotated", Nothing) -> problem "invalid-value" (L.render (L.child path "offset")) "A rotated order needs an offset"
            _ -> problem "invalid-value" (L.render (L.child path "order")) "Expected declared, reversed or rotated; only rotated takes an offset"

collectionCheck :: U.UseContract -> Execution -> Check ()
collectionCheck contract plan = issues (E.collectionProblems contract (collection plan))
  where
    issues [] = value ()
    issues (found : rest) = problem "invalid-value" (E.collectionAt found) (E.collectionMessage found) *> issues rest

schedules :: Natural -> Natural -> Execution -> Check ()
schedules inputs executions plan =
    counted *> traverse unchanged (E.unchanged inputs (paired plan) (repeats plan)) *> value ()
  where
    counted
        | fromIntegral (length (repeats plan)) + 1 == executions = value ()
        | otherwise = problem "invalid-value" "execution:repeats" ("The contract requires " ++ show executions ++ " candidate executions, so the plan needs " ++ show (executions - 1) ++ " repeats")
    unchanged position = problem "schedule-unchanged" ("execution:repeats[" ++ show position ++ "]") "This repeat has the same request order and batch partition as the paired run or another repeat"
