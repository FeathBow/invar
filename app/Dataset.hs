{-# LANGUAGE OverloadedStrings #-}

module Dataset (Identity (..), Cycle, decode, instantiate, sessions) where

import Data.ByteString (ByteString)
import Invar.Cohort qualified as C
import Invar.Infer qualified as I
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
instantiate selected workload = do
    tasks <- traverse prepare (Workload.tasks workload)
    either (Left . show) Right (C.withCohort (C.Definition (policy selected) tasks) (const ()))
    pure (L.Cycle tasks (Workload.order workload) (Workload.delivery workload))
  where
    prepare sample = do
        planned <- either (Left . show) Right (I.prepare I.Request {I.artifact = policy selected, I.tokenizer = tokenizer selected, I.base = base selected, I.assembly = assembly selected, I.prompt = Workload.prompt sample, I.tokens = Workload.tokens sample, I.temperature = Workload.temperature sample, I.seed = Workload.seed sample})
        pure (C.Task (Workload.name sample) (Workload.group sample) planned (Workload.rule sample))

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
