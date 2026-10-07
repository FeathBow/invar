{-# LANGUAGE OverloadedStrings #-}

module Dataset (Identity (..), Cycle, decode, instantiate, sessions) where

import Data.ByteString (ByteString)
import Invar.Loop qualified as L
import Invar.Workload qualified as Workload

type Cycle = Workload.Cycle
data Identity = Identity {policy :: String, tokenizer :: String, base :: String, assembly :: String}

decode :: Identity -> ByteString -> Either String [Cycle]
decode selected encoded = do
    cycles <- Workload.cycles <$> Workload.decode encoded
    mapM_ (instantiate selected) cycles
    pure cycles

instantiate :: Identity -> Cycle -> Either String L.Cycle
instantiate selected = L.instantiate (policy selected, tokenizer selected, base selected, assembly selected)

sessions :: Maybe String -> Either String [[(String, String)]]
sessions Nothing = Right [[]]
sessions (Just listed) = do
    let devices = split listed
    if null devices || any null devices || any (any (`elem` (", " :: String))) devices
        then Left "Invalid --devices: expected comma-separated nonempty device identifiers"
        else Right [[("CUDA_VISIBLE_DEVICES", device)] | device <- devices]
  where
    split text = case break (== ',') text of
        (item, []) -> [item]
        (item, _ : rest) -> item : split rest
