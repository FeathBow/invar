{-# LANGUAGE GHC2021 #-}

module UpdateReport (inspect) where

import Data.ByteString (ByteString)
import Invar.Learn.Report qualified as Report
import Numeric.Natural (Natural)

inspect :: Natural -> ByteString -> Either String String
inspect call encoded = Report.logDigest <$> Report.admit call encoded
