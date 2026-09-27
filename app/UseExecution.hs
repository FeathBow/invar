module UseExecution (Execution (..), Side (..), decode, schedules) where

import Check (Check, andThen, problem, value)
import Data.Aeson (Object, Value (..))
import Invar.Use.Execution qualified as E
import Located qualified as L
import Numeric.Natural (Natural)

data Side = Side {worker :: FilePath, configuration :: FilePath, adapter :: FilePath}

data Execution = Execution {backend :: String, python :: FilePath, cache :: FilePath, reference :: Side, candidate :: Side, paired :: E.Arrangement, repeats :: [E.Arrangement]}

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
               )
            <* (L.field "invalid-value" top fields "collection" `andThen` collection)
  where
    top = L.root "execution"
    text fields name = L.field "invalid-value" top fields name `andThen` L.text "invalid-value" (L.child top name)
    formatted fields = text fields "format" `andThen` \found -> if found == "invar-use-execution-v1" then value () else problem "invalid-value" (L.render (L.child top "format")) "Expected invar-use-execution-v1"
    supported found = if found == "mlx" then value found else problem "unsupported" (L.render (L.child top "backend")) "Only the mlx backend can run a contract"
    side fields name =
        let sides = L.child top "sides"
            path = L.child sides name
         in L.field "invalid-value" top fields "sides" `andThen` L.object sides `andThen` \entries ->
                L.field "invalid-value" sides entries name `andThen` L.object path `andThen` \entry ->
                    L.only path ["worker", "configuration", "adapter"] entry
                        *> (Side <$> named path entry "worker" <*> named path entry "configuration" <*> named path entry "adapter")
    named path entry name = L.field "invalid-value" path entry name `andThen` L.text "invalid-value" (L.child path name)
    collection found =
        let path = L.child top "collection"
         in L.object path found `andThen` \fields ->
                L.only path ["scores", "full_vocabulary_steps"] fields
                    *> (L.field "invalid-value" path fields "scores" `andThen` L.boolean (L.child path "scores") `andThen` \scores -> if scores then problem "unsupported" (L.render (L.child path "scores")) "Scored paths are not collected" else value ())
                    *> (L.field "invalid-value" path fields "full_vocabulary_steps" `andThen` L.list (L.child path "full_vocabulary_steps") `andThen` \steps -> if null steps then value () else problem "unsupported" (L.render (L.child path "full_vocabulary_steps")) "Full-vocabulary probes are not collected")

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

schedules :: Natural -> Natural -> Execution -> Check ()
schedules inputs executions plan =
    counted *> traverse unchanged (E.unchanged inputs (paired plan) (repeats plan)) *> value ()
  where
    counted
        | fromIntegral (length (repeats plan)) + 1 == executions = value ()
        | otherwise = problem "invalid-value" "execution:repeats" ("The contract requires " ++ show executions ++ " candidate executions, so the plan needs " ++ show (executions - 1) ++ " repeats")
    unchanged position = problem "schedule-unchanged" ("execution:repeats[" ++ show position ++ "]") "This repeat has the same request order and batch partition as the paired run or another repeat"
