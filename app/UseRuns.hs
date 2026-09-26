module UseRuns (observe) where

import Data.Aeson (FromJSON, Value, eitherDecodeStrict, parseJSON, toJSON)
import Data.Aeson.Types (parseEither)
import Data.ByteString qualified as Bytes
import Invar.Use qualified as U
import Numerical qualified

observe :: FilePath -> U.UseContract -> IO (Either String U.Observed)
observe path contract = do
    decoded <- eitherDecodeStrict <$> Bytes.readFile path
    case decoded of
        Left problem -> pure (Left problem)
        Right entries -> case traverse shape entries of
            Left problem -> pure (Left problem)
            Right shaped -> do
                cases <- traverse build shaped
                pure (either (Left . show) Right (U.observe (U.BoundRun (U.declaredDomain contract) (U.declaredMeasurement contract) cases)))
  where
    shape :: [Value] -> Either String (U.Key, Value, Value)
    shape entry = case entry of
        [cohort, name, arguments] -> keyed cohort name arguments (toJSON ([] :: [[String]]))
        [cohort, name, arguments, repeated] -> keyed cohort name arguments repeated
        _ -> Left "Each run entry is [cohort index, input key, paired-observation argument array] with an optional array of repeated candidate argument arrays"
    keyed cohort name arguments repeated = do
        key <- U.Key <$> decode cohort <*> decode name
        pure (key, arguments, repeated)
    build (key, arguments, repeated) = do
        paired <- either fail pure (decode arguments) >>= Numerical.readPair
        repeats <- either fail pure (decode repeated) >>= traverse Numerical.readRun
        pure (U.Case key paired repeats)
    decode :: (FromJSON value) => Value -> Either String value
    decode = parseEither parseJSON
