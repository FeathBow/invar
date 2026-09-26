module Located (Path, root, child, item, render, field, optionalField, text, natural, positive, rational, object, list, boolean, only) where

import Check (Check, andThen, problem, value)
import Data.Aeson (Object, Value (..), parseJSON)
import Data.Aeson.Key qualified as Key
import Data.Aeson.KeyMap qualified as Fields
import Data.Aeson.Types (parseMaybe)
import Data.Char (isSpace)
import Data.Foldable (toList)
import Data.List (intercalate)
import Data.Text qualified as Text
import Numeric.Natural (Natural)
import UsePlan (readRational)

data Path = Path String [String]

root :: String -> Path
root label = Path label []

child :: Path -> String -> Path
child (Path label segments) name = Path label (segments ++ [name])

item :: Path -> Int -> Path
item (Path label segments) index = Path label (initial ++ [final ++ "[" ++ show index ++ "]"])
  where
    (initial, final) = case reverse segments of
        last' : rest -> (reverse rest, last')
        [] -> ([], "")

render :: Path -> String
render (Path label segments) = label ++ ":" ++ intercalate "." segments

field :: String -> Path -> Object -> String -> Check Value
field code path fields name = case Fields.lookup (Key.fromString name) fields of
    Just Null -> problem code (render (child path name)) (name ++ " is required")
    Just found -> value found
    Nothing -> problem code (render (child path name)) (name ++ " is required")

optionalField :: Object -> String -> Maybe Value
optionalField fields name = case Fields.lookup (Key.fromString name) fields of
    Just Null -> Nothing
    found -> found

text :: String -> Path -> Value -> Check String
text code path (String found)
    | Text.all isSpace found = problem code (render path) "A nonblank text is required"
    | otherwise = value (Text.unpack found)
text _ path _ = problem "invalid-value" (render path) "Expected text"

natural :: Path -> Value -> Check Natural
natural path found = case parseMaybe parseJSON found :: Maybe Integer of
    Just whole | whole >= 0 -> value (fromInteger whole)
    _ -> problem "invalid-value" (render path) "Expected a natural number"

positive :: Path -> Value -> Check Natural
positive path found = natural path found `andThen` \number -> if number > 0 then value number else problem "invalid-value" (render path) "Expected a positive number"

rational :: Path -> Value -> Check Rational
rational path (String found) = maybe (problem "invalid-value" (render path) "Expected an exact rational such as 1/40 or a decimal such as 0.025") value (readRational (Text.unpack found))
rational path _ = problem "invalid-value" (render path) "Expected an exact rational written as text, such as \"1/40\""

object :: Path -> Value -> Check Object
object _ (Object fields) = value fields
object path _ = problem "invalid-value" (render path) "Expected an object"

list :: Path -> Value -> Check [Value]
list _ (Array values) = value (toList values)
list path _ = problem "invalid-value" (render path) "Expected a list"

boolean :: Path -> Value -> Check Bool
boolean _ (Bool found) = value found
boolean path _ = problem "invalid-value" (render path) "Expected true or false"

only :: Path -> [String] -> Object -> Check ()
only path allowed fields = case [name | name <- map Key.toString (Fields.keys fields), name `notElem` allowed] of
    [] -> value ()
    extra -> problem "invalid-value" (render path) ("Unexpected fields: " ++ intercalate ", " extra)
