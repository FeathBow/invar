{-# LANGUAGE OverloadedStrings #-}

module Dataset (Identity (..), Cycle, decode, instantiate, sessions) where

import Control.Monad (unless, when)
import Data.Aeson (Object, Value, eitherDecodeStrict', withObject, (.:))
import Data.Aeson.Key (Key)
import Data.Aeson.KeyMap qualified as Map
import Data.Aeson.Types (Parser, parseEither)
import Data.ByteString (ByteString)
import Data.List (sort)
import Invar.Cohort qualified as C
import Invar.Infer qualified as I
import Invar.Loop qualified as L
import Invar.Reward qualified as Reward
import Invar.Schedule qualified as Schedule
import Numeric.Natural (Natural)

data Sample = Sample {name :: String, group :: String, prompt :: String, tokens :: Natural, temperature :: Double, seed :: Integer, rule :: Reward.Rule}
data Cycle = Cycle [Sample] [Natural] [Natural]
data Identity = Identity {policy :: String, tokenizer :: String, base :: String, assembly :: String}

decode :: Identity -> ByteString -> Either String [Cycle]
decode selected encoded = do
    values <- eitherDecodeStrict' encoded
    cycles <- traverse (parseEither parseCycle) values
    when (null cycles) (Left "Expected a nonempty sequence of training cycles")
    mapM_ (instantiate selected) cycles
    pure cycles

parseCycle :: Value -> Parser Cycle
parseCycle = withObject "Training cycle" $ \fields -> do
    exact ["tasks", "order", "delivery"] fields
    tasks <- fields .: "tasks" >>= traverse parseSample
    order <- fields .: "order"
    delivery <- fields .: "delivery"
    _ <- either (fail . show) pure (Schedule.prepare (fromIntegral (length tasks)) order delivery)
    pure (Cycle tasks order delivery)

parseSample :: Value -> Parser Sample
parseSample = withObject "Training sample" $ \fields -> do
    exact ["name", "group", "prompt", "tokens", "temperature", "seed", "answer"] fields
    expected <- fields .: "answer"
    reward <- either (fail . show) pure (Reward.decimal expected)
    Sample <$> fields .: "name" <*> fields .: "group" <*> fields .: "prompt" <*> fields .: "tokens" <*> fields .: "temperature" <*> fields .: "seed" <*> pure reward

exact :: [Key] -> Object -> Parser ()
exact expected fields = unless (sort (Map.keys fields) == sort expected) (fail "Unexpected or missing training fields")

instantiate :: Identity -> Cycle -> Either String L.Cycle
instantiate selected (Cycle samples order delivery) = do
    tasks <- traverse prepare samples
    either (Left . show) Right (C.withCohort (C.Definition (policy selected) tasks) (const ()))
    pure (L.Cycle tasks order delivery)
  where
    prepare sample = do
        planned <- either (Left . show) Right (I.prepare I.Request {I.artifact = policy selected, I.tokenizer = tokenizer selected, I.base = base selected, I.assembly = assembly selected, I.prompt = prompt sample, I.tokens = tokens sample, I.temperature = temperature sample, I.seed = seed sample})
        pure (C.Task (name sample) (group sample) planned (rule sample))

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
