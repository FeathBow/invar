module Main (main) where

import Data.Ratio (denominator, numerator)
import Invar.Numerical.Distribution (Bounds (..), enclose)
import Text.Read (readMaybe)

main :: IO ()
main = getContents >>= mapM_ (putStrLn . evaluate) . lines

evaluate :: String -> String
evaluate line = case readMaybe line of
    Nothing -> "{\"error\":\"invalid test input\"}"
    Just (left, right) -> case enclose left right of
        Left problem -> "{\"error\":" ++ show problem ++ "}"
        Right (forward, backward) -> "[" ++ render forward ++ "," ++ render backward ++ "]"

render :: Bounds -> String
render InfiniteKL = "{\"kind\":\"positive_infinity\"}"
render (FiniteBounds lower upper) = "{\"kind\":\"finite\",\"lower\":" ++ ratio lower ++ ",\"upper\":" ++ ratio upper ++ "}"
  where
    ratio value = show [numerator value, denominator value]
