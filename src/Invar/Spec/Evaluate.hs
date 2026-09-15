{-# LANGUAGE Safe #-}

module Invar.Spec.Evaluate (
    World,
    Semantics (..),
    Emission (..),
    Error (..),
    evaluate,
    prepareCommands,
    runCommands,
) where

import Control.Monad (foldM, unless)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set (Set)
import Data.Set qualified as Set
import Invar.Spec.Dependency qualified as D
import Invar.Spec.Operator (Operator, OperatorError)
import Invar.Spec.Operator qualified as Operator
import Invar.Spec.Program
import Invar.Spec.Value (Scalar (..), Value (..))
import Numeric.Natural (Natural)

type World = Map Source (Value Natural)

data Semantics = Semantics {schema :: Schema, meanings :: Map String Operator}
    deriving (Eq, Show)

data Emission = Emission {sinkName :: String, sinkSpecification :: String, payload :: Value Natural}
    deriving (Eq, Show)

data Error
    = InvalidProgram D.Error
    | MissingInput Source
    | InvalidInput Source Type
    | ExtraInputs (Set Source)
    | MissingMeaning String
    | MeaningMismatch String Signature Signature
    | OperatorFailure String OperatorError
    | EvaluationInvariant String
    deriving (Eq, Show)

data Binding = Datum (Value Natural) | BoundKey Natural

data Context = Context
    { semantics :: Semantics
    , world :: World
    , bindings :: Map String Binding
    }

evaluate :: Semantics -> World -> Expr -> Either Error (Value Natural)
evaluate meaning inputs expression = do
    _ <- either (Left . InvalidProgram) Right (D.analyze (schema meaning) expression)
    context <- prepare meaning inputs
    eval context expression

runCommands :: Semantics -> World -> [Command] -> Either Error [Emission]
runCommands meaning inputs commands = do
    execute <- either (Left . InvalidProgram) Right (prepareCommands meaning commands)
    execute inputs

-- The returned function binds this checked program; every call still checks its world.
prepareCommands :: Semantics -> [Command] -> Either D.Error (World -> Either Error [Emission])
prepareCommands meaning commands = do
    _ <- D.checkCommands (schema meaning) commands
    pure execute
  where
    execute inputs = do
        context <- prepare meaning inputs
        traverse (emit context) commands
    emit context (Emit name spec expression) = Emission name spec <$> eval context expression

prepare :: Semantics -> World -> Either Error Context
prepare meaning inputs = do
    let declared = sources (schema meaning)
        extra = Map.keysSet inputs Set.\\ Map.keysSet declared
    unless (Set.null extra) (Left (ExtraInputs extra))
    mapM_ validate (Map.toList declared)
    pure (Context meaning inputs Map.empty)
  where
    validate (source, kind) = case Map.lookup source inputs of
        Nothing -> Left (MissingInput source)
        Just value -> unless (matches kind value) (Left (InvalidInput source kind))

eval :: Context -> Expr -> Either Error (Value Natural)
eval context expression = case expression of
    Variable name -> lookupDatum context name
    Read source -> readInput context source
    Constant _ value -> Right value
    Fields fields -> Record <$> traverse (eval context) fields
    Project value name -> eval context value >>= project name
    If condition first second -> do
        predicate <- eval context condition >>= boolean
        eval context (if predicate then first else second)
    Let name value rest -> do
        evaluated <- eval context value
        eval (extend context [(name, Datum evaluated)]) rest
    Primitive name inputs -> traverse (eval context) inputs >>= primitive context name
    Collect collection -> collect context collection
    KeyEqual first second -> do
        left <- lookupKey context first
        right <- lookupKey context second
        pure (Atom (Boolean (left == right)))

lookupDatum :: Context -> String -> Either Error (Value Natural)
lookupDatum context name = case Map.lookup name (bindings context) of
    Just (Datum value) -> Right value
    _ -> Left (EvaluationInvariant ("Missing payload binding: " ++ name))

lookupKey :: Context -> String -> Either Error Natural
lookupKey context name = case Map.lookup name (bindings context) of
    Just (BoundKey key) -> Right key
    _ -> Left (EvaluationInvariant ("Missing key binding: " ++ name))

readInput :: Context -> ReadSource -> Either Error (Value Natural)
readInput context input = case input of
    Input source -> value source
    Random source -> value source
  where
    value source = maybe (Left (MissingInput source)) Right (Map.lookup source (world context))

project :: String -> Value Natural -> Either Error (Value Natural)
project name (Record fields) = maybe (Left (EvaluationInvariant ("Missing field: " ++ name))) Right (Map.lookup name fields)
project _ _ = Left (EvaluationInvariant "Projection operand is not a record")

boolean :: Value Natural -> Either Error Bool
boolean (Atom (Boolean value)) = Right value
boolean _ = Left (EvaluationInvariant "Condition is not Boolean")

primitive :: Context -> String -> [Value Natural] -> Either Error (Value Natural)
primitive context name values = do
    let meaning = semantics context
    implementation <- maybe (Left (MissingMeaning name)) Right (Map.lookup name (meanings meaning))
    declared <- maybe (Left (EvaluationInvariant ("Missing primitive declaration: " ++ name))) Right (Map.lookup name (primitives (schema meaning)))
    let actual = Operator.signature implementation
    unless (actual == declared) (Left (MeaningMismatch name declared actual))
    either (Left . OperatorFailure name) Right (Operator.apply implementation values)

extend :: Context -> [(String, Binding)] -> Context
extend context additions = context {bindings = Map.fromList additions <> bindings context}

mapScopeBindings :: MapBody -> (Natural, Value Natural) -> [(String, Binding)]
mapScopeBindings binder (key, value) = [(keyName binder, BoundKey key), (valueName binder, Datum value)]

mapEntries :: Value Natural -> Either Error [(Natural, Value Natural)]
mapEntries (Mapping values) = Right (Map.toAscList values)
mapEntries _ = Left (EvaluationInvariant "Traversal operand is not a map")

sequenceItems :: Value Natural -> Either Error [Value Natural]
sequenceItems (Sequence values) = Right values
sequenceItems _ = Left (EvaluationInvariant "Traversal operand is not a sequence")

collect :: Context -> Collection -> Either Error (Value Natural)
collect context collection = case collection of
    MapValues input binder -> do
        entries <- eval context input >>= mapEntries
        Mapping . Map.fromDistinctAscList <$> traverse (transform context binder) entries
    FoldMap fold -> mapFold context fold
    FoldSequence fold -> sequenceFold context fold

transform :: Context -> MapBody -> (Natural, Value Natural) -> Either Error (Natural, Value Natural)
transform context binder entry@(key, _) = do
    value <- eval (extend context (mapScopeBindings binder entry)) (mapExpression binder)
    pure (key, value)

mapFold :: Context -> MapFold -> Either Error (Value Natural)
mapFold context fold = do
    entries <- eval context (mapInput fold) >>= mapEntries
    seed <- eval context (mapInitial fold)
    foldM advance seed entries
  where
    binder = mapScope fold
    advance accumulator entry =
        let additions = mapScopeBindings binder entry ++ [(mapAccumulator fold, Datum accumulator)]
         in eval (extend context additions) (mapExpression binder)

sequenceFold :: Context -> SequenceFold -> Either Error (Value Natural)
sequenceFold context fold = do
    items <- eval context (sequenceInput fold) >>= sequenceItems
    seed <- eval context (sequenceInitial fold)
    foldM advance seed items
  where
    advance accumulator item =
        let additions = [(itemName fold, Datum item), (sequenceAccumulator fold, Datum accumulator)]
         in eval (extend context additions) (sequenceExpression fold)
