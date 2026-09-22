{-# LANGUAGE GHC2021 #-}

module ScoreFact (forge) where

import Invar.Score qualified as S

-- Reject: [GHC-76037]
forge = S.Fact
