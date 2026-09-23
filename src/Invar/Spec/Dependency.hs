{-# LANGUAGE Safe #-}

module Invar.Spec.Dependency (
    Analysis (..),
    Error (..),
    analyze,
    checkSchema,
    checkCommands,
) where

import Control.Monad (unless)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set (Set)
import Data.Set qualified as Set
import Invar.Spec.Program

data Analysis = Analysis {valueType :: Type, dependencies :: Set Source}
    deriving (Eq, Show)

data Error
    = MissingSource Source
    | MissingVariable String
    | KeyEscapes String
    | NotKey String
    | NotRandom Source
    | MissingField String
    | ExpectedRecord Type
    | ExpectedMap Type
    | ExpectedSequence Type
    | TypeMismatch Type Type
    | KeyCarryingLiteral Type
    | InvalidLiteral Type
    | MissingPrimitive String
    | KeyCarryingPrimitive String
    | ArgumentTypes [Type] [Type]
    | DuplicateBinder String
    | MissingSink String
    | WrongSpecification String String
    | UnrecordedSources (Set Source)
    | ForbiddenSources (Set Source)
    deriving (Eq, Show)

data Binding = Payload Analysis | BoundKey (Set Source)

data Scope = Scope {declarations :: Schema, bindings :: Map String Binding}

data FoldCheck = FoldCheck
    { foldScope :: Scope
    , accumulator :: String
    , body :: Expr
    , initial :: Analysis
    , range :: Analysis
    }

analyze :: Schema -> Expr -> Either Error Analysis
analyze schema = infer (Scope schema Map.empty)

infer :: Scope -> Expr -> Either Error Analysis
infer scope expression = case expression of
    Variable name -> variable scope name
    Read source -> readSource scope source
    Constant kind value -> do
        unless (keyFree kind) (Left (KeyCarryingLiteral kind))
        unless (matches kind value) (Left (InvalidLiteral kind))
        pure (Analysis kind Set.empty)
    Fields fields -> do
        values <- traverse (infer scope) fields
        pure (combine (RecordType (fmap valueType values)) (Map.elems values))
    Project value name -> infer scope value >>= project name
    If condition first second -> conditional scope (condition, first, second)
    Let name value rest -> do
        checked <- infer scope value
        infer scope {bindings = Map.insert name (Payload checked) (bindings scope)} rest
    Primitive name inputs -> primitive scope name inputs
    Collect collection -> collect scope collection
    KeyEqual first second -> do
        left <- key scope first
        right <- key scope second
        pure (Analysis BooleanType (left <> right))

lookupBinding :: Scope -> String -> Either Error Binding
lookupBinding scope name = maybe (Left (MissingVariable name)) Right (Map.lookup name (bindings scope))

variable :: Scope -> String -> Either Error Analysis
variable scope name = do
    bound <- lookupBinding scope name
    case bound of
        Payload value -> Right value
        BoundKey _ -> Left (KeyEscapes name)

key :: Scope -> String -> Either Error (Set Source)
key scope name = do
    bound <- lookupBinding scope name
    case bound of
        BoundKey deps -> Right deps
        Payload _ -> Left (NotKey name)

readSource :: Scope -> ReadSource -> Either Error Analysis
readSource scope input = case input of
    Input source -> declared source
    Random source@(LogicalRandom _) -> declared source
    Random source -> Left (NotRandom source)
  where
    declared source = case Map.lookup source (sources (declarations scope)) of
        Nothing -> Left (MissingSource source)
        Just kind -> Right (Analysis kind (Set.singleton source))

combine :: Type -> [Analysis] -> Analysis
combine kind values = Analysis kind (Set.unions (map dependencies values))

expect :: Type -> Analysis -> Either Error ()
expect kind value = unless (kind == valueType value) (Left (TypeMismatch kind (valueType value)))

project :: String -> Analysis -> Either Error Analysis
project name value = case valueType value of
    RecordType fields -> case Map.lookup name fields of
        Nothing -> Left (MissingField name)
        Just kind -> Right value {valueType = kind}
    kind -> Left (ExpectedRecord kind)

conditional :: Scope -> (Expr, Expr, Expr) -> Either Error Analysis
conditional scope (condition, first, second) = do
    predicate <- infer scope condition
    expect BooleanType predicate
    left <- infer scope first
    right <- infer scope second
    expect (valueType left) right
    pure (combine (valueType left) [predicate, left, right])

primitive :: Scope -> String -> [Expr] -> Either Error Analysis
primitive scope name expressions = do
    signature <- maybe (Left (MissingPrimitive name)) Right (Map.lookup name (primitives (declarations scope)))
    unless (all keyFree (result signature : arguments signature)) (Left (KeyCarryingPrimitive name))
    values <- traverse (infer scope) expressions
    let actual = map valueType values
    unless (actual == arguments signature) (Left (ArgumentTypes (arguments signature) actual))
    pure (combine (result signature) values)

extend :: Scope -> [(String, Binding)] -> Either Error Scope
extend scope additions = do
    local <- unique Map.empty additions
    pure scope {bindings = local <> bindings scope}
  where
    unique collected [] = Right collected
    unique collected ((name, binding) : rest)
        | Map.member name collected = Left (DuplicateBinder name)
        | otherwise = unique (Map.insert name binding collected) rest

mapBindings :: Analysis -> MapBody -> Either Error [(String, Binding)]
mapBindings input binder = case valueType input of
    MapType item -> Right [(keyName binder, BoundKey (dependencies input)), (valueName binder, Payload input {valueType = item})]
    kind -> Left (ExpectedMap kind)

collect :: Scope -> Collection -> Either Error Analysis
collect scope collection = case collection of
    MapValues source binder -> do
        input <- infer scope source
        local <- mapBindings input binder >>= extend scope
        value <- infer local (mapExpression binder)
        pure (combine (MapType (valueType value)) [input, value])
    FoldMap fold -> mapFold scope fold
    FoldSequence fold -> sequenceFold scope fold

mapFold :: Scope -> MapFold -> Either Error Analysis
mapFold scope fold = do
    input <- infer scope (mapInput fold)
    seed <- infer scope (mapInitial fold)
    bound <- mapBindings input (mapScope fold)
    local <- extend scope (bound ++ [(mapAccumulator fold, Payload seed)])
    foldResult
        FoldCheck
            { foldScope = local
            , accumulator = mapAccumulator fold
            , body = mapExpression (mapScope fold)
            , initial = seed
            , range = input
            }

sequenceFold :: Scope -> SequenceFold -> Either Error Analysis
sequenceFold scope fold = do
    input <- infer scope (sequenceInput fold)
    seed <- infer scope (sequenceInitial fold)
    item <- case valueType input of
        SequenceType kind -> Right input {valueType = kind}
        kind -> Left (ExpectedSequence kind)
    local <- extend scope [(itemName fold, Payload item), (sequenceAccumulator fold, Payload seed)]
    foldResult
        FoldCheck
            { foldScope = local
            , accumulator = sequenceAccumulator fold
            , body = sequenceExpression fold
            , initial = seed
            , range = input
            }

foldResult :: FoldCheck -> Either Error Analysis
foldResult checked = do
    let scope = foldScope checked
        seed = initial checked
        base = dependencies seed <> dependencies (range checked)
        acc = Payload seed {dependencies = base}
        local = scope {bindings = Map.insert (accumulator checked) acc (bindings scope)}
    value <- infer local (body checked)
    expect (valueType seed) value
    pure seed {dependencies = base <> dependencies value}

checkSchema :: Schema -> Either Error ()
checkSchema schema = do
    mapM_ signature (Map.toList (primitives schema))
    mapM_ (validateSink schema) (Map.elems (sinks schema))
  where
    signature (name, declared) =
        unless (all keyFree (result declared : arguments declared)) (Left (KeyCarryingPrimitive name))

checkCommands :: Schema -> [Command] -> Either Error [Analysis]
checkCommands schema = traverse (checkCommand schema)

checkCommand :: Schema -> Command -> Either Error Analysis
checkCommand schema (Emit name spec expression) = do
    sink <- maybe (Left (MissingSink name)) Right (Map.lookup name (sinks schema))
    unless (spec == specification sink) (Left (WrongSpecification (specification sink) spec))
    validateSink schema sink
    value <- analyze schema expression
    expect (inputType sink) value
    let forbidden = dependencies value Set.\\ allowed sink
    unless (Set.null forbidden) (Left (ForbiddenSources forbidden))
    pure value

validateSink :: Schema -> Sink -> Either Error ()
validateSink schema sink = do
    let unrecorded = Set.filter operational (allowed sink) Set.\\ recorded sink
    unless (Set.null unrecorded) (Left (UnrecordedSources unrecorded))
    mapM_ declared (Set.toList (allowed sink <> recorded sink))
  where
    declared source = unless (Map.member source (sources schema)) (Left (MissingSource source))
    operational (Operational _) = True
    operational _ = False
