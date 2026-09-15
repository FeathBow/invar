{-# LANGUAGE GHC2021 #-}

module Report (inspect) where

import Data.ByteString (ByteString)
import Invar.Evaluation qualified as Evaluation
import Invar.Workload qualified as Workload

inspect :: ByteString -> Evaluation.Run -> ByteString -> Either String [Evaluation.Sample]
inspect tasks selected report = do
    expected <- Workload.decode tasks
    observed <- Evaluation.admit expected selected report
    pure (Evaluation.samples observed)
