{-# LANGUAGE GHC2021 #-}

module Measurement (inspect) where

import Data.Aeson (encode)
import Data.ByteString.Lazy (ByteString)
import Invar.Measurement qualified as Measurement
import Invar.Workload qualified as Workload

inspect :: Measurement.Source -> Workload.Document -> (String, FilePath) -> IO ByteString
inspect source tasks selected = encode <$> Measurement.admit source tasks selected
