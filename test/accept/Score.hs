{-# LANGUAGE GHC2021 #-}

module Score (inspect, probe) where

import Data.ByteString (ByteString)
import Invar.Score qualified as S
import Numeric.Natural (Natural)

inspect :: S.Call -> Int -> ByteString -> Either S.Error S.LogRatio
inspect call status encoded = S.logRatio <$> S.admit call status encoded

probe :: [Natural] -> S.Plan -> Either S.Error S.Plan
probe = S.withProbe
