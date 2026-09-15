module NativeCodec (select, options) where

import Control.Monad (unless)
import Data.Maybe (isNothing)
import Invar.Learn.State qualified as State
import Options qualified as O
import System.Console.GetOpt (OptDescr)

select :: O.Fields -> Either String State.Decoder
select fields = case O.optional fields "codec-mode" of
    Nothing -> external
    Just "process" -> external
    Just "stdio" -> do
        unless (isNothing (O.optional fields "python") && isNothing (O.optional fields "codec")) (Left "Stdio codec mode does not accept external decoder options")
        pure State.Standard
    _ -> Left "Unknown native codec mode"
  where
    external = State.External <$> O.required fields "python" <*> O.required fields "codec"

options :: [OptDescr (String, String)]
options = O.descriptions [("python", "Native serialization Python executable"), ("codec", "Native serialization script"), ("codec-mode", "process (default) or stdio for a hosting serialization bridge")]
