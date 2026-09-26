module Check (Check, problem, value, located, andThen, run) where

import Envelope (Problem (..))

data Check result = Check [Problem] (Maybe result)

instance Functor Check where
    fmap transform (Check problems result) = Check problems (transform <$> result)

instance Applicative Check where
    pure result = Check [] (Just result)
    Check left transform <*> Check right result = Check (left ++ right) (transform <*> result)

problem :: String -> String -> String -> Check result
problem code at message = Check [Problem code at message] Nothing

value :: result -> Check result
value = pure

located :: Either Problem result -> Check result
located = either (\found -> Check [found] Nothing) pure

run :: Check result -> Either [Problem] result
run (Check [] (Just result)) = Right result
run (Check problems _) = Left problems

infixl 1 `andThen`

andThen :: Check result -> (result -> Check next) -> Check next
andThen (Check problems (Just result)) next = let Check more final = next result in Check (problems ++ more) final
andThen (Check problems Nothing) _ = Check problems Nothing
